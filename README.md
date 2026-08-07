# Day 4 — RAG

![CI](https://github.com/m-hamzaj/rag/actions/workflows/ci.yml/badge.svg)

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
# or, a browser dashboard instead of the CLI:
docker compose up -d ui             # http://localhost:8082
```

`ingest` and `tests` need no API key at all — only `ask`/`ui` (answer
generation) do. The `python -m rag.cli ...` in the `ask` commands is
required in full — `docker compose run` replaces a service's default
command rather than appending to it.

## The interface (`docker compose up -d ui`)

A small Streamlit page at `:8082` for asking questions without a
terminal: a text box, an Ask button, the answer, and its source links.
It's a thin wrapper around `rag/ask.py` — the same function the CLI
calls — not a second implementation of retrieval; `ui/app.py` only calls
into `rag/` and renders what comes back.

The one thing it adds beyond the CLI: a "Why this answer" panel showing
the actual chunks retrieval pulled back and their real similarity scores,
so a refusal (or a shaky answer) is inspectable instead of a black box —
directly useful given `SIMILARITY_THRESHOLD` is the real refusal
mechanism (see "How retrieval actually works" below), not something to
take on faith.

Verified live: real question, real Chroma retrieval, real Groq call, via
Streamlit's own `AppTest` harness against the actual running index —
correct answer, correct citation, and the retrieved-chunks panel showing
the same similarity scores as the CLI and the standalone verification
below.

**Two more things on the page**, both there because "I don't know what to
ask" turned out to be the real usability problem, not a retrieval bug:

- **A sidebar corpus browser** — every indexed article's title, grouped
  by source, with a filter box. Answers only ever come from what's
  actually indexed; a refusal on a question about something genuinely
  absent (jaguars in Brazil, say — checked for real, the corpus has zero
  articles on it, only a passing citation-list mention and an unrelated
  "not a jaguar, it's a jaguarundi" aside) is correct behavior, not a bug,
  but a user has no way to know that without seeing what's actually
  there. Backed by a new `db.list_documents()`, deduped from chunk
  metadata.
- **An "Add new articles to ask about" panel** — this project only
  *answers* questions, it doesn't scrape (that's Day 3's job). Adding an
  article is still two steps — push it into Day 1 via Day 3's crawler
  dashboard (`localhost:8080`, "Crawl a URL") or `docker compose run --rm
  crawler`, then click "Index new articles" here — but the second step no
  longer needs a terminal, and it no longer means waiting on a full
  corpus rebuild for one new article. `rag/ingest.py`'s new
  `ingest_new_documents()` fetches Day 1's documents, diffs against
  `list_documents()`'s URLs, and only chunks/embeds what's actually new.
  Verified live end to end: pushed one real, previously-unindexed Audubon
  article through Day 3's `crawl_urls()`, clicked "Index new articles"
  (reported "Indexed 1 new article(s) into 1 chunks" — not 125), then
  asked a question about that specific article and got a correct,
  correctly-cited answer.

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
   (`db.search_similar_chunks`), and sort them into two tiers:
   `accepted` (clears `SIMILARITY_THRESHOLD` — strong enough to answer
   directly) and `related` (below that but at or above the lower
   `RELATED_SIMILARITY_THRESHOLD` — topically close, not a direct match).
5. **Answer, answer with a caveat, or refuse** (`ask.py`, `generate.py`):
   - `accepted` chunks exist → answered directly, citing `[N]` markers,
     via the normal prompt.
   - No `accepted` chunks, but `related` ones exist → answered via a
     different prompt that permits general/adjacent information *only
     from those chunks*, on the condition the answer says plainly it
     isn't a direct match. The reply is prefixed
     `*Related, not a direct answer:*` in code, not left to the model to
     remember to say so.
   - Neither → `"I don't know."`, LLM never called.
   In both the direct and related cases, the LLM can still refuse on its
   own if the chunks it was actually given turn out not to be useful —
   this is what stops the related tier from fabricating a connection
   just to seem helpful.
6. **Citations** (`generate.py`, `ask.py`) — the answer's `[N]` markers
   are parsed back out and mapped to the real chunks they point to (falls
   back to citing every retrieved chunk if the model used no markers at
   all — better to over-cite than silently drop attribution). Citations
   are then deduped to one entry per source article, and dropped entirely
   if the model's own answer was itself a refusal.

**Why a related tier at all.** A flat "I don't know." is honest but
throws away real, on-topic corpus content whenever a question is close to
the corpus's subject matter without being a direct hit — a real, reported
usability problem: on-topic questions about animals/plants/conservation
were getting refused even when the corpus had adjacent information worth
surfacing. The fix keeps the anti-fabrication guarantee intact (nothing
is invented; both prompts still require every claim to trace back to an
excerpt, and the related prompt explicitly forbids inventing specifics to
sound more relevant than the sources actually are) while giving the
reader *something* instead of nothing whenever the corpus has anything
adjacent at all.

**Verified against the real corpus and the real model, not just asserted.**
Probed real questions against the live 1,485-chunk collection: most
on-topic questions already land in `accepted` (this corpus turns out to
cover its subject matter broadly — e.g. "How do jaguars in Brazil avoid
human conflict?" scores 0.55), so the related tier is a genuine minority
case, not the common path. One that lands in it: *"How does the axolotl
regenerate its limbs?"* — best match 0.334, below `SIMILARITY_THRESHOLD`
but above `RELATED_SIMILARITY_THRESHOLD`. The 5 related chunks retrieved
were about seed germination, growing kale, deep-sea species, animal
eyeshine, and swift fox reintroduction — genuinely not useful for this
question. Real answer produced:

> *Related, not a direct answer:* The sources don't directly cover this,
> but they do provide some general information about biology and the
> natural world. [...] Overall, these excerpts don't offer any specific
> information about axolotl limb regeneration, and a more direct source
> would be needed to answer this question.

This is the honest failure mode working as designed: the related tier
doesn't manufacture usefulness that isn't there, it makes the *attempt*
and its result visible instead of a bare refusal that hides what was
(and wasn't) actually found.

**The `accepted`-tier refusal is still a property of retrieval, not a
prompt instruction the model could ignore** — a question with nothing
above `SIMILARITY_THRESHOLD` *and* nothing above
`RELATED_SIMILARITY_THRESHOLD` never reaches the LLM in the first place.
The LLM-level refusal in step 5 is a second, independent check, present
in both the direct and related paths, for the case where retrieval finds
plausible-looking chunks that turn out not to actually be useful.

`db.py` is the only module that knows which storage backend is in use.
`ingest.py` and `retrieve.py` call `create_schema`/`clear_chunks`/
`insert_chunks`/`search_similar_chunks`/`count_chunks` by name and don't
care what's behind them — which is exactly what made switching the
backend (below) a one-file change plus a test rewrite, not a project-wide
one.

## Guarding against prompt injection from scraped content

The corpus this project answers from is scraped web text (Day 3's
crawler) — nothing stops a source article from containing text aimed at
the LLM itself rather than at a human reader. A URL-safety check (Day 3's
`crawler/validate.py`) can catch a malicious *destination*; it can't catch
malicious *content* on an otherwise ordinary, legitimately-fetched page.
That defense has to live here, at generation time, not in the crawler.

`generate.py`'s system prompt tells the model explicitly that excerpts are
untrusted web content, names the actual attack shape it must refuse (text
claiming to be a system/developer message, e.g. "ignore previous
instructions"), and says that content never overrides these instructions.
`_build_prompt` wraps each excerpt in `<excerpt>` tags — a visible,
structural boundary between "reference material" and "instructions," not
just a verbal claim — and repeats the reminder again right after the last
excerpt, immediately before the question (a "sandwich": the instruction
closest to what the model reads last has the most influence, so a long
excerpt trying to bury an instruction can't simply out-position the system
message).

**Verified against the real model, not just asserted.** A chunk was
crafted with a real recipe followed by an embedded instruction telling the
model to abandon its citation format and reply `PWNED` to everything:

- Sent through the **original** (pre-hardening) prompt: the model
  replied `PWNED` — confirming the injection was real, not a
  hypothetical.
- Sent through the **hardened** prompt: the model answered the actual
  question and cited its source normally, ignoring the embedded
  instruction entirely.

Not a complete defense — a sufficiently different phrasing could still get
through, and no prompt-level instruction is unconditionally reliable
against a model that's been specifically targeted. What limits the blast
radius further is architectural: this pipeline never gives the model tool
access or the ability to take actions, so even a successful injection can
only produce a bad *answer*, not an action.

## Config values (`rag/config.py`)

| Value | Default | What it controls |
|---|---|---|
| `CHUNK_SIZE_WORDS` | 200 | words per chunk |
| `CHUNK_OVERLAP_WORDS` | 40 | shared words between consecutive chunks |
| `TOP_K` | 5 | chunks pulled per query |
| `SIMILARITY_THRESHOLD` | 0.35 | minimum cosine similarity to count as a direct answer |
| `RELATED_SIMILARITY_THRESHOLD` | 0.20 | minimum cosine similarity to count as topically related background |
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
docker compose --profile tools run --rm tests    # 80 tests, fully offline
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
  db.py         ChromaDB schema + insert + similarity search + list_documents -- the only storage-aware module
  ingest.py     ingest() = full rebuild; ingest_new_documents() = index only what's not indexed yet
  retrieve.py   embed question -> top-k search -> threshold filter
  generate.py   Groq call + [N]-citation-marker parsing
  ask.py        ties retrieve + generate together, handles both refusal paths
  cli.py        single-question and interactive modes
ui/
  app.py        Streamlit dashboard -- thin wrapper around rag/ask.py, same logic as the CLI
tests/          66 offline tests
```
