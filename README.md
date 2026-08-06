# Day 4 — RAG

Question answering over Day 3's 125 scraped nature/wildlife/gardening
articles. Chunks them, embeds the chunks, stores the vectors in ChromaDB.
A question comes in, gets embedded, pulls the top matching chunks, and
gets answered from those chunks only — with citations, or an honest
refusal when nothing in the corpus is actually relevant.

No LangChain, no LlamaIndex. Chunking, embedding, retrieval, and citation
parsing are all hand-written in `rag/` — see "How retrieval actually
works" below for the real loop.

**On the storage choice:** the brief that started this project specified
Postgres with pgvector, and that's what was built and verified first (a
committed run: 125 documents → 1,484 chunks, real questions answered with
citations, a real refusal on an out-of-corpus question). It was switched
to ChromaDB afterward on the team's decision. Functionally either is
fine for a corpus this size — the tradeoff is pgvector living in the same
Postgres this project set already runs elsewhere (joins against
relational data, one less moving part) versus Chroma's simpler standalone
setup. Both the switch and the reasoning are logged here rather than
silently overwriting what the brief actually asked for. See "Storage:
switched from pgvector to Chroma" below for what changed and what was
found re-verifying it.

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
docker compose up -d db            # ChromaDB
docker compose run --rm ingest     # chunk + embed + store every Day 1 article
docker compose run --rm ask python -m rag.cli "your question"
# or, interactively:
docker compose run --rm ask python -m rag.cli
```

`ingest` and `tests` need no API key at all — only `ask` (answer
generation) does. The `python -m rag.cli ...` in the `ask` commands is
required in full — `docker compose run` replaces a service's default
command rather than appending to it.

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
3. **Store** (`db.py`, `ingest.py`) — one entry per chunk in a Chroma
   collection explicitly configured for cosine distance: the chunk text,
   its embedding, and which article (URL + title) it came from, as
   metadata. `ingest.py` pages through Day 1's `/documents` (100
   items/request max), chunks and embeds each one, and upserts by an ID
   derived from `(document_url, chunk_index)` so re-ingesting doesn't
   duplicate.
4. **Retrieve** (`retrieve.py`) — embed the question with the same model,
   pull the `TOP_K` most similar chunks by cosine similarity
   (`db.search_similar_chunks`), and keep only the ones clearing
   `SIMILARITY_THRESHOLD`.
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

`db.py` is the only module that knows which storage backend is in use.
`ingest.py` and `retrieve.py` call `create_schema`/`clear_chunks`/
`insert_chunks`/`search_similar_chunks`/`count_chunks` by name and don't
care what's behind them — which is exactly what made switching the
backend (below) a one-file change plus a test rewrite, not a project-wide
one.

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

## Storage: switched from pgvector to Chroma

`db.py` was rewritten against `chromadb`'s `HttpClient`, keeping every
function name and signature from the pgvector version so nothing above
it changed. Two real things came up doing this for real, not just editing
code:

**Chroma's default distance metric isn't cosine.** It's squared L2 unless
the collection says otherwise, which would have silently made
`SIMILARITY_THRESHOLD = 0.35` meaningless — a value tuned against real
cosine similarity scores compared against a completely different scale.
Fixed by creating the collection with `metadata={"hnsw:space": "cosine"}`
explicitly (`db.py`'s `_COLLECTION_METADATA`), and by keeping the same
`1 - distance = similarity` convention the pgvector version used, so
`retrieve.py`'s threshold comparison needed zero changes.

**The official `chromadb/chroma` server image has no Python, curl, or
wget in it.** Chroma's server is a Rust binary now, not the Python app an
initial `docker-compose.yml` healthcheck assumed
(`python -c "import urllib.request; ..."`) — the container came up but
Docker reported it `unhealthy` every time, `exec`-ing in and running
`which curl wget python3` turned up nothing. `bash` *is* present, so the
healthcheck now opens a raw TCP connection via `bash`'s `/dev/tcp`
pseudo-device instead of an HTTP request — a good enough proxy for "the
server is listening" without needing a real HTTP client inside the image.

**Re-verification, not just re-running tests:** after the switch, the
full real pipeline was re-run end to end — ingest again produced 125
documents → 1,484 chunks (identical to the pgvector run), and both real
questions from "Real verification" below were asked again against the
Chroma-backed index. The answers and citations came back identical, and
the similarity scores in the threshold table below are byte-for-byte the
same as the pgvector run — expected, since both compute exact cosine
similarity over the same embeddings; neither backend approximates at
this corpus size.

Historical note, kept for anyone who ends up back on pgvector: the
original build hit a real bug where pgvector's `<=>` operator rejected a
bare Python list (`psycopg.errors.UndefinedFunction: operator does not
exist: vector <=> double precision[]`) — `register_vector()` only adapts
a `pgvector.Vector` or numpy array into the `vector` type, not a plain
list, and a bare list happened to still work on `INSERT` because the
target column's declared type gave Postgres a coercion hint a raw
comparison doesn't get. Not relevant to the current Chroma-backed code,
but a real trap if this ever moves back.

## Real verification

125 documents → **1,484 chunks**, on both the original pgvector run and
the re-verified Chroma run.

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
docker compose --profile tools run --rm tests    # 61 tests, fully offline
```

Chunking (boundaries, overlap, no dropped words, config validation),
embedding (batch/single, empty input, lazy model loading — the ONNX model
itself is mocked, no real inference in the test suite), Chroma storage
(cosine-space collection creation, upsert-by-derived-id, distance→
similarity conversion, empty-result handling — all against a mocked
client/collection, no real Chroma server needed to run the suite),
ingest (pagination through Day 1, chunk↔embedding zipping, full-rebuild
wipe), retrieval (threshold filtering, top-k passthrough), generation
(prompt building, `[N]`-marker parsing including out-of-range and
no-marker fallback, the real Groq request shape), and `ask()`'s two
refusal paths (no chunks retrieved; LLM refusing on its own) — all via
monkeypatched HTTP/DB/model, no real network call, no API key needed.

## Project layout

```
rag/
  config.py     chunk size, top-k, similarity threshold, model names -- all env-overridable
  chunk.py      word-based sliding-window chunker
  embed.py      fastembed wrapper (local, no API key)
  db.py         ChromaDB schema + insert + similarity search -- the only storage-aware module
  ingest.py     pulls every Day 1 document, chunks + embeds + stores it
  retrieve.py   embed question -> top-k search -> threshold filter
  generate.py   Groq call + [N]-citation-marker parsing
  ask.py        ties retrieve + generate together, handles both refusal paths
  cli.py        single-question and interactive modes
tests/          60 offline tests
```
