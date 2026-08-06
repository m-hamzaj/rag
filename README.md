# Day 4 — RAG

Question answering over Day 3's 125 scraped nature/wildlife/gardening
articles. Chunks them, embeds the chunks, stores the vectors in Postgres
(pgvector). A question comes in, gets embedded, pulls the top matching
chunks, and gets answered from those chunks only — with citations, or an
honest refusal when nothing in the corpus is actually relevant.

No LangChain, no LlamaIndex. Chunking, embedding, retrieval, and citation
parsing are all hand-written in `rag/` — see "How retrieval actually
works" below for the real loop.

## Prerequisites

Day 1 (Documents API) and Day 3 (crawler) must already have run — this
project reads Day 1's `/documents` endpoint, it doesn't scrape anything
itself.

```bash
cd ../documents-api/documents-api && docker compose up -d   # Day 1
# Day 3's articles should already be pushed into it
```

Then, from this project:

```bash
cp .env.example .env               # fill in GROQ_API_KEY
docker compose up -d db            # pgvector Postgres
docker compose run --rm ingest     # chunk + embed + store every Day 1 article
docker compose run --rm ask python -m rag.cli "your question"
# or, interactively:
docker compose run --rm ask python -m rag.cli
```

`ingest` and `tests` need no API key at all — only `ask` (answer
generation) does.

## How retrieval actually works

No framework retriever, no vector-store abstraction — this is the whole
loop, in `rag/`:

1. **Chunk** (`chunk.py`) — a word-based sliding window over each
   article's text: `CHUNK_SIZE_WORDS` words per chunk, advancing by
   `CHUNK_SIZE_WORDS - CHUNK_OVERLAP_WORDS` each step, so consecutive
   chunks share `CHUNK_OVERLAP_WORDS` words of context. Both are config
   values (`rag/config.py`), not literals — the brief calls this out
   specifically, since they're the first things worth sweeping once real
   retrieval quality is in front of you.
2. **Embed** (`embed.py`) — `fastembed` running
   `sentence-transformers/all-MiniLM-L6-v2` locally over ONNX (not
   sentence-transformers/torch directly — same model weights, no torch
   dependency, smaller image). Free, no API key, no network at query
   time. 384-dim vectors.
3. **Store** (`db.py`, `ingest.py`) — one row per chunk in a pgvector
   `chunks` table: the chunk text, its embedding, and which article
   (URL + title) it came from. `ingest.py` pages through Day 1's
   `/documents` (100 items/request max), chunks and embeds each one, and
   upserts on `(document_url, chunk_index)` so re-ingesting doesn't
   duplicate.
4. **Retrieve** (`retrieve.py`) — embed the question with the same model,
   pull the `TOP_K` most similar chunks by cosine similarity
   (`db.search_similar_chunks`, brute-force `ORDER BY embedding <=> ...
   LIMIT`), and keep only the ones clearing `SIMILARITY_THRESHOLD`.
5. **Refuse, or generate** (`ask.py`, `generate.py`) — if nothing clears
   the threshold, the answer is `"I don't know."` and the LLM is never
   called at all. Otherwise the surviving chunks go to Groq
   (`llama-3.3-70b-versatile`) with a prompt that numbers each excerpt and
   instructs the model to cite `[N]` markers for whatever it actually
   used, and to say `"I don't know."` itself if the excerpts don't
   actually answer the question.
6. **Citations** (`generate.py`, `ask.py`) — the answer's `[N]` markers
   are parsed back out and mapped to the real chunks they point to (falls
   back to citing every retrieved chunk if the model used no markers at
   all — better to over-cite than silently drop attribution). Citations
   are then deduped to one entry per source article, and dropped entirely
   if the model's own answer was itself a refusal.

The refusal is a property of retrieval, not a prompt instruction the
model could ignore — a question with nothing above `SIMILARITY_THRESHOLD`
never reaches the LLM in the first place. The LLM-level refusal in step 5
is a second, independent check for the case where retrieval finds
plausible-looking chunks that turn out not to actually answer the
question.

## Config values (`rag/config.py`)

| Value | Default | What it controls |
|---|---|---|
| `CHUNK_SIZE_WORDS` | 200 | words per chunk |
| `CHUNK_OVERLAP_WORDS` | 40 | shared words between consecutive chunks |
| `TOP_K` | 5 | chunks pulled per query |
| `SIMILARITY_THRESHOLD` | 0.35 | minimum cosine similarity to count as real evidence |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | fastembed model |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | generation model |

All env-overridable, none hardcoded into `chunk.py`/`retrieve.py` directly.

## Two real bugs, found by actually running it

**pgvector's `<=>` operator doesn't accept a bare Python list.**
`register_vector()` (from `pgvector.psycopg`) teaches psycopg how to
adapt a `pgvector.Vector` — or a numpy array — into the database's
`vector` type. It does *not* cover a plain Python `list[float]`. The
first real query after a real ingest failed outright:

```
psycopg.errors.UndefinedFunction: operator does not exist: vector <=> double precision[]
```

A bare list got sent as a generic `double precision[]`, and Postgres has
no `<=>` between that and `vector`. What made this easy to miss: the
*ingest* step's `INSERT`s had already succeeded with bare lists, because
the target column's declared type (`VECTOR(384)`) gives Postgres a
coercion hint an `INSERT` can use that a raw comparison operator can't.
1,484 chunks went in clean; the very first real question then broke on
read. Fixed by wrapping every embedding — both on insert and on query —
in `pgvector.Vector(...)` explicitly in `db.py`, rather than relying on
an implicit coercion that only worked for one of the two paths. Covered
by regression tests in `tests/test_db.py` asserting the wrapped type is
actually what gets sent.

**Day 1's database had a stray, off-topic document.** Before running the
real ingest, a document titled *"Claude Code Cost Optimization: Cut Your
Token Spend"* from `claudedirectory.org` turned up in Day 1 — not from
any of Day 3's 5 configured nature sources, and not something this
project's own tooling had pushed. Day 1's API is just a local service on
`localhost:8000`; anything on the machine can write to it. Caught by
diffing every stored document's `source` field against the 5 known
domains before ingesting, rather than trusting the row count alone.
Removed via Day 1's own `DELETE /documents/{id}` before the real ingest
ran, so the RAG corpus is nature/wildlife/gardening content only.

## Real verification

125 documents → **1,484 chunks** on the committed ingest run.

**A real question, answered correctly with a citation:**

> **Q: How do you make hummingbird nectar?**
> To make hummingbird nectar, you need to mix sugar and boiling water
> until the sugar is dissolved, then cool and fill the feeder [1][2]. The
> specific ingredients are 1/4 cup refined white sugar and 1 cup boiling
> water [2].
>
> Sources: *How to Make Hummingbird Nectar* (audubon.org)

**A second, from a different source:**

> **Q: What is being done to protect tigers in Nepal?**
> According to [1], efforts are being made in Nepal to protect tigers,
> including habitat conservation and management, poaching control,
> conservation of prey species, and biological corridor conservation. Bed
> Kumar Dhakal, spokesperson for the Department of National Parks and
> Wildlife Conservation, mentions that these efforts are showing positive
> outcomes in the field of tiger conservation.
>
> Sources: *Nepal's newest national parks drive tiger recovery, but new
> concerns arise* (news.mongabay.com)

**A real refusal:**

> **Q: What is the capital of France?**
> I don't know.

No citations, and the LLM was never called — retrieval's top match
scored 0.229, below the 0.35 threshold.

**The threshold, checked against real similarity scores, not assumed:**

| Question | Top similarity | In corpus? |
|---|---|---|
| How do you make hummingbird nectar? | 0.700 | yes |
| What is being done to protect tigers in Nepal? | (retrieved, answered) | yes |
| What is the capital of France? | 0.229 | no |
| How do I fix a Python ImportError? | 0.200 | no |
| What is the best way to invest in cryptocurrency? | 0.303 | no |

Every on-topic question tested scored 0.6+; every off-topic one capped
around 0.2–0.3. `SIMILARITY_THRESHOLD = 0.35` sits in the real gap
between those two clusters, not a guessed number — this is the first
value worth re-checking if `TOP_K` or the chunking config change later
this week, since a different chunk size shifts what "typical" similarity
looks like.

## Tests

```bash
docker compose --profile tools run --rm tests    # 60 tests, fully offline
```

Chunking (boundaries, overlap, no dropped words, config validation),
embedding (batch/single, empty input, lazy model loading — the ONNX model
itself is mocked, no real inference in the test suite), pgvector storage
(schema creation never touches `register_vector` — the chicken-and-egg
regression from an earlier build — insert/search both wrap embeddings in
`Vector`, upsert-on-conflict), ingest (pagination through Day 1, chunk↔
embedding zipping, full-rebuild wipe), retrieval (threshold filtering,
top-k passthrough), generation (prompt building, `[N]`-marker parsing
including out-of-range and no-marker fallback, the real Groq request
shape), and `ask()`'s two refusal paths (no chunks retrieved; LLM
refusing on its own) — all via monkeypatched HTTP/DB/model, no real
network call, no API key needed to run the suite.

## Project layout

```
rag/
  config.py     chunk size, top-k, similarity threshold, model names -- all env-overridable
  chunk.py      word-based sliding-window chunker
  embed.py      fastembed wrapper (local, no API key)
  db.py         pgvector schema + insert + similarity search
  ingest.py     pulls every Day 1 document, chunks + embeds + stores it
  retrieve.py   embed question -> top-k search -> threshold filter
  generate.py   Groq call + [N]-citation-marker parsing
  ask.py        ties retrieve + generate together, handles both refusal paths
  cli.py        single-question and interactive modes
tests/          60 offline tests
```
