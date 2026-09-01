# Day 4–7 — RAG, evaluating it, tuning retrieval, then an agent

![CI](https://github.com/m-hamzaj/rag/actions/workflows/ci.yml/badge.svg)

Question answering over Day 3's 125 scraped nature/wildlife/gardening
articles. Day 4–6 are framework-free — chunking, embedding, retrieval,
and citation parsing are all hand-written in `rag/`. Day 7's agent is the
one deliberate exception (LangChain, per updated requirements) — see
below for why.

- **Day 4** builds the pipeline: chunk → embed → store in ChromaDB →
  retrieve top matches → answer with citations, or refuse honestly when
  nothing in the corpus is relevant.
- **Day 5** measures it against a hand-written 20-question eval set
  instead of guessing at quality.
- **Day 6** adds keyword and hybrid retrieval alongside vector search,
  sweeps chunk size and retrieval mode against that same eval set, and
  picks a winner with evidence.
- **Day 7** builds a tool-calling agent for questions one retrieval pass
  can't answer, and measures where it actually helps — and where it's
  worse than Day 4–6's plain pipeline.

All four live in one project because Day 5–7 test Day 4's code directly
(`eval.py`/`eval_agent.py` import `rag/`, same as the CLI and UI). Full
eval numbers and reasoning: **`RESULTS.md`**. Deep dives, bug stories,
and verification transcripts: **`NOTES.md`**.

## Running it

Day 1 (Documents API) and Day 3 (crawler) must already have run — this
project reads Day 1's `/documents` endpoint, it doesn't scrape itself.

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
# or, a browser dashboard instead of the CLI:
docker compose up -d ui             # http://localhost:8082
```

`ingest` and `tests` need no API key — only `ask`/`ui` do. After changing
anything under `rag/`, run `docker compose build` (no service name) —
each service builds its own image from the same `Dockerfile`, so
rebuilding one doesn't rebuild the others. See `NOTES.md` for the bug
this caused once.

## How it works

1. **Chunk** (`chunk.py`) — packs each article along paragraph/markdown-
   header boundaries, up to `CHUNK_SIZE_WORDS` words, with
   `CHUNK_OVERLAP_WORDS` shared between consecutive chunks.
2. **Embed** (`embed.py`) — `fastembed` running `all-MiniLM-L6-v2`
   locally, no API key, no network at query time.
3. **Store** (`db.py`) — ChromaDB, explicitly configured for cosine
   similarity.
4. **Retrieve** (`retrieve.py`) — top `TOP_K` chunks by `RETRIEVAL_MODE`
   (`vector`, `keyword`/BM25, or `hybrid`/RRF fusion of both), sorted
   into `accepted` (clears `SIMILARITY_THRESHOLD`) or `related` (clears
   the lower `RELATED_SIMILARITY_THRESHOLD`) tiers.
5. **Answer** (`ask.py`, `generate.py`) — `accepted` chunks → direct
   cited answer; only `related` chunks → answered with a "related, not a
   direct answer" caveat; neither → `"I don't know."`, LLM never called.
   The LLM can also refuse on its own if the chunks it got aren't
   actually useful.

Full walkthrough with the reasoning behind each choice, plus the prompt-
injection defense and the pgvector→Chroma migration story: `NOTES.md`.

## The agent (`rag/agent.py`)

Plain RAG above does one search, then answers. For questions that need
evidence from more than one article — a comparison, a count, a "find X
then look up Y" chain — `rag/agent.py` gives the model three tools
(`search_articles`, `read_article`, `finish`) and loops: call the model,
run whichever tool it picked, feed the result back, repeat until
`finish` or a hard limit stops it. Three hard limits, enforced in code:
max steps, max cost (checked against real token usage after every call),
and duplicate-call loop detection — all three produce an honest,
evidence-based fallback answer instead of nothing.

Built on LangChain (`ChatGroq` + tool-calling), the one place this
project uses a framework — a deliberate, later change from an initial
hand-rolled version, not the project's default stance. `eval_agent.py`
measures it against plain RAG on both the original 20 questions and 10
new ones written specifically to need multiple retrieval hops. Full
numbers, and several real infrastructure bugs found running it live
(a request-size limit, a misconfigured `tool_choice`, rate-limit
mislabeling): `RESULTS.md`'s Day 7 section.

## Config values (`rag/config.py`)

| Value | Default | What it controls |
|---|---|---|
| `CHUNK_SIZE_WORDS` | 200 | words per chunk |
| `CHUNK_OVERLAP_WORDS` | 40 | shared words between consecutive chunks |
| `TOP_K` | 5 | chunks pulled per query |
| `SIMILARITY_THRESHOLD` | 0.35 | minimum cosine similarity to count as a direct answer |
| `RELATED_SIMILARITY_THRESHOLD` | 0.33 | minimum cosine similarity to count as topically related background |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | fastembed model |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | generation model |
| `RETRIEVAL_MODE` | `vector` | `vector` \| `keyword` \| `hybrid` |

All env-overridable, none hardcoded.

## Results

- **Day 5 baseline:** 18/20 (90%) Answer correct, 15/16 (94%) Top-1 and
  Top-5.
- **Day 6 winner:** `chunk=200`, `vector` — same config as Day 5's
  baseline, confirmed against 6 real runs (3 chunk sizes × vector/hybrid)
  as the most accurate, and cheaper/faster than the alternative that beat
  it in neither cost nor speed.
- **Day 7 agent vs. plain RAG:** on the original 20 questions (a clean,
  rate-limit-free run), the agent scores 13/20 (65%) vs. plain RAG's
  18/20 (90%), at 4.4x the cost and 9x the latency — it loses on the easy
  set, as expected. On 10 new questions written specifically to need
  multiple retrieval hops, plain RAG fails all 10; the agent succeeds
  where live testing let it run uninterrupted.

Both the Day 5/6 numbers and the Day 7 comparison carry real uncertainty
at n=20/n=10 questions (a single flipped question moves the score 5–10
points) — `RESULTS.md` covers this precisely, next to the numbers
themselves, along with the full tuning history, the Day 6 sweep, a
chunking fix made in response to external review, and the Day 7 agent
build.

## Tests

```bash
docker compose --profile tools run --rm tests    # 144 tests, fully offline
```

Covers chunking, embedding, storage, ingest, retrieval (vector/keyword/
hybrid), generation, citations, both refusal paths, and the Day 7 agent
loop (tool dispatch, all three hard limits, duplicate-call detection,
error-type handling) — all against mocked HTTP/DB/model, no real network
call or API key needed.

## Project layout

```
rag/
  config.py           chunk size, top-k, thresholds, retrieval mode, model names -- all env-overridable
  chunk.py            paragraph/markdown-section-aware chunker (word-count slide as fallback)
  embed.py            fastembed wrapper (local, no API key)
  db.py               ChromaDB schema + insert + similarity search + list_documents
  ingest.py           ingest() = full rebuild; ingest_new_documents() = index only what's new
  keyword_search.py   hand-rolled BM25 index, no external library
  retrieve.py         embed question -> vector/keyword/hybrid ranking -> threshold filter
  generate.py         Groq call + [N]-citation-marker parsing + token usage
  ask.py              ties retrieve + generate together, handles both refusal paths
  agent.py            Day 7 -- LangChain tool-calling agent loop, hard limits enforced in code
  cli.py              single-question and interactive modes
ui/
  app.py              Streamlit dashboard -- thin wrapper around rag/ask.py
tests/                144 offline tests
eval.py                Day 5/6 -- live eval against the real corpus, see RESULTS.md
eval_agent.py          Day 7 -- agent vs. plain RAG comparison harness, see RESULTS.md
data/agent_eval_set.json  Day 7 -- 10 multi-step questions plain RAG can't answer
```
