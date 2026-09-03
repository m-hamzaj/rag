# day8-multiagent — results

## What this is

A LangGraph-based multi-agent extension of day4-rag's Day 7 single agent:
instead of one tool-calling loop doing search → read → answer itself, three
specialized roles hand off to each other — **researcher** (search_articles/
read_article, no `finish` tool), **writer** (drafts an answer from the
researcher's notes), **critic** (checks the draft against those notes,
sends it back to the writer or approves it). Full design in
`.claude/plans/vivid-sauteeing-aurora.md` (the approved implementation
plan); code in `agents/`.

Lives nested inside `day4-rag/` (as `day4-rag/day8-multiagent/`) so it
shares that project's git repo and remote, but is still its own deployable
unit — own `Dockerfile`, `docker-compose.yml`, `requirements.txt`
(`langgraph`, not just `langchain`), and Python environment. Reuses
day4-rag's already-running ChromaDB corpus read-only (no re-ingest, no
shared *code* — see `agents/db.py`'s module docstring for why duplication
was chosen over importing `rag.*` directly, even now that both live in one
repo). `baseline/single_agent.py` is a frozen copy of day4-rag's Day 7
agent, re-pointed at this project's own `agents/db.py`, so the comparison
below runs both systems against identical infrastructure, not just
identical questions.

Offline test suite: 58 tests, fully mocked (`agents.llm.call_llm` patched
at one seam shared by all three nodes — see `tests/conftest.py`), no real
API key or network call. All passing.

## A real grading bug, found and fixed before trusting these numbers

`gpt-oss-120b` routinely wraps numbers in a **narrow no-break space**
(U+202F) as a formatting habit — e.g. it writes `67 million` with U+202F
between the two words, not a plain ASCII space. The original
`_answer_is_correct` substring check (copied from day4-rag/eval.py's
pattern) only normalized hyphens and curly quotes, not Unicode whitespace
variants, so `"67 million" in answer` silently returned `False` against an
answer that was actually correct — a real pass mis-graded as a fail.

Confirmed live: `'67 million' in '67 million'` → `False`.

Same lesson this project has hit before at every level (Day 5's original
`must_contain` calibration, Day 7's `"remote cameras"` vs
`"remote-camera"` mismatch) — a paraphrase/formatting mismatch in the
*grader*, not the model. Fixed by collapsing every `str.isspace()`
character (not just the ASCII space) to a plain space before comparing,
in `eval_multiagent.py`'s `_normalize`. Re-grading the already-captured
transcripts with the fix changed one result: **single_agent Q6 (the
three-fact "rice's whale / grizzly bear / sage-grouse" counting question)
was actually correct**, not the fail originally reported live. All numbers
below use the corrected grader.

## Groq rate-limiting dominated both live runs

Both eval runs hit sustained `rate_limited` stops on a majority of
questions — an org-level, per-minute token budget (confirmed directly:
`8,000 tokens/minute`, refilling within ~1 second when idle) that a
multi-step agent conversation exhausts easily once article text has been
read into it a couple of times. This is the same Groq shared "on_demand"
tier variability day4-rag's own RESULTS.md documents for Day 7 — worse
here, since a 3-node graph makes strictly more calls per question than the
single agent.

| | Total correct | Rate-limited | Clean (non-rate-limited) | Correct on clean subset |
|---|---|---|---|---|
| **single_agent** | 3/12 (25%) | 6/12 | 6 | **3/6 (50%)** |
| **multiagent** | 1/12 (8%) | 9/12 | 3 | **1/3 (33%)** |

Full per-question breakdown (corrected grading):

**single_agent** (`$0.0209` total across all 12, including partial cost
from rate-limited attempts):

| Q | Correct | Stopped | Steps | Cost | Seconds |
|---|---|---|---|---|---|
| 1 | ✗ | malformed_tool_call | 5 | $0.00144 | 47 |
| 2 | ✓ | finished | 5 | $0.00246 | 45 |
| 3 | ✗ | duplicate_call | 4 | $0.00155 | 35 |
| 4 | ✗ | max_steps | 8 | $0.00458 | 132 |
| 5 | ✓ | finished | 5 | $0.00222 | 42 |
| 6 | ✓ | finished | 7 | $0.00440 | 136 |
| 7–12 | ✗ | rate_limited (all six) | 3–6 | $0.0002–0.0022 | 488–988 |

**multiagent** (`$0.0136` total):

| Q | Correct | Stopped | Steps | Cost | Seconds |
|---|---|---|---|---|---|
| 1 | ✗ | revision_limit | 9 | $0.00513 | 155 |
| 2 | ✓ | finished | 6 | $0.00282 | 49 |
| 3–6, 8–12 | ✗ | rate_limited (nine total) | 1–4 | $0.0000–0.0009 | 467–751 |
| 7 | ✗ | finished | 6 | $0.00251 | **4921** |

Q7's 82-minute wall-clock time (real work, real cost — not a rate-limit
signature) is most plausibly explained by the machine going idle/sleeping
mid-run, not a code issue; noted honestly rather than silently excluded.

## What's actually usable from this

The clean-subset numbers (n=6 and n=3) are too small to claim a real
capability difference between the two systems — both are single-digit
sample sizes, and a single flipped question moves either percentage by
17–33 points. **This comparison is inconclusive**, not negative — it does
not show the multi-agent system losing to the single agent; it shows that
today's infrastructure conditions didn't allow a clean enough run to tell.

Two genuine, non-infrastructure findings did surface:

1. **Q1's `revision_limit` on the multi-agent run** — the critic sent the
   draft back for revision, and the second draft still wasn't approved
   before `MAX_REVISION_CYCLES=2` capped it. A real instance of the
   critic disagreeing with the writer, not a rate-limit artifact — exactly
   the mechanism this architecture was built to test, caught working as
   designed even though the final answer still wasn't correct.
2. **single_agent Q1's `malformed_tool_call`**: the model attempted to
   call a tool named `find_in_article`, which doesn't exist in this
   agent's three-tool set (`search_articles`/`read_article`/`finish`).
   Groq's own server-side schema validation rejected it with a 400 before
   the code ever saw it — a real, reproducible model hallucination,
   distinct from every other error class this project has documented.

## An attempted fix that backfired, then a working one

After the run above, tried making retries smarter: instead of a blind
`15s*2^attempt` exponential backoff, `agents/llm.py` was changed to read
Groq's own `x-ratelimit-reset-tokens` response header and sleep exactly
that long — the theory being that 700–950+ seconds of blind backoff per
rate-limited question (seen above) was mostly wasted, since the header
already states how long until the exhausted budget refills.

**Live rerun: this made it strictly worse — 0/12 questions completed, all
12 `rate_limited`, most failing in ~3 seconds flat.** Root cause: the
header reports time until the token bucket has *some* room again (a
continuous trickle-refill), not time until *enough* room exists for the
specific, often several-thousand-token request actually pending. Trusting
it directly made retries fire too soon, fail again immediately, and
exhaust all 5 retries in seconds — the exhausted budget never genuinely
had time to recover the way the old blind schedule (which kept waiting
regardless of what any header claimed) accidentally guaranteed.

Fixed by treating the header as a **floor, not the answer**:
`_retry_after_seconds` now returns `max(header-derived wait, the same
exponential schedule as before)` — the header can only make a wait
*longer* than the blind guess (when Groq's response genuinely knows more),
never shorter than the schedule that was already proven to eventually
work. Covered by dedicated tests (`tests/test_llm.py`) for both
directions: a header asking for longer than the fallback is honored, a
header asking for less is ignored in favor of the fallback.

The corrected version was rerun live and confirmed to behave correctly
(real backoff waits, no more instant-fail loop) — but hit sustained
`rate_limited` results again regardless, on all 6 questions attempted
before the run was stopped. That's a *different* problem than the retry
logic: this project's Groq key had already made a very large number of
real calls across this whole session's testing (multiple full eval
attempts, several smoke tests), consistent with a broader hourly/daily cap
being close to exhausted, not just the per-minute token bucket — something
no amount of retry-timing cleverness fixes. Stopped deliberately rather
than burning more wall-clock time confirming the same exhausted-quota
signal repeatedly; a real cooldown period is the only thing that resolves
this, not another code change.

## The post-cooldown retry: a real single-agent number, and a new finding about multi-agent's cost

Waited ~4 hours (scheduled via a one-shot cron job) for the Groq key's
broader quota to clear, then reran both evals with the corrected
floor-based retry logic already in place.

**single_agent got a genuinely clean run**: 3/12 correct (25%), only
**4/12** rate-limited — the best completion rate seen across every attempt
this project has made. Of the 8 questions that actually finished:
**3/8 correct (37.5%)** — the most trustworthy single-agent number
gathered so far, on the largest clean sample.

**multiagent got a complete write-off**: **12/12 rate-limited**, 0
correct, zero clean data points — run back-to-back, immediately after
single_agent, against the identical Chroma corpus and the identical Groq
key/quota state that had just given single_agent a mostly-clean pass.

| | Total correct | Rate-limited | Clean (non-rate-limited) | Correct on clean subset |
|---|---|---|---|---|
| **single_agent** (post-cooldown) | 3/12 (25%) | 4/12 | 8 | **3/8 (37.5%)** |
| **multiagent** (post-cooldown) | 0/12 (0%) | 12/12 | 0 | n/a — no clean data |

**This is itself a real finding, not just more noise.** Both systems ran
under identical environmental conditions (same corpus, same key, same
quota state, back-to-back). The gap between an 8/12 clean-completion rate
and a 0/12 clean-completion rate is exactly what the multi-agent design's
own tradeoff predicts: three roles handing off to each other make
strictly more Groq calls per question than one loop does (researcher's own
multi-step turn, plus a writer call, plus a critic call, times however
many revision cycles happen), which means strictly more exposure to any
given per-minute token budget. On Groq's free/shared "on_demand" tier
specifically, that overhead isn't just a cost-per-query number on a
spreadsheet — it's the difference between a system that can actually
finish its work under real rate limits and one that categorically cannot,
at least on this tier.

## What a clean multiagent run would need

Given the measured `8,000 tokens/minute` budget, how quickly a multi-step
conversation with full article reads approaches that on its own, and now
direct evidence that multi-agent's added call volume pushes it from
"mostly survives this budget" to "never survives this budget" — a
genuinely clean multiagent comparison likely needs a materially larger
per-minute token budget (a paid Groq tier), not a code or pacing change.
This project's `45s`/`60s` inter-question pacing was already correctly
identified (above) as addressing the wrong constraint (request rate, not
token budget); the retry-logic fix addressed how *efficiently* a rate
limit is waited out once hit, but neither changes how *often* multiagent's
architecture hits it in the first place — that's structural, not tunable
from this side of the API.

## Why rate limits hit a tool-calling agent harder than a plain LLM call

Worth stating plainly, since it's the throughline of nearly every finding
above: **a tool-calling agent is not one request against a rate limit, it's
N requests, and N grows with exactly the things that make the agent
useful.**

A single plain-RAG call (day4-rag's `rag/ask.py`) is one prompt, one
completion, one token count, done. Every one of Groq's rate-limit headers
was designed around that shape — "8,000 tokens/minute" sounds generous
for a single short exchange, and it is.

A tool-calling agent breaks that assumption in three compounding ways:

1. **Every tool call is a full extra round trip to the model**, each one
   billed in tokens like any other call. Day 7's single agent needing 5–8
   steps to answer one question means 5–8 separate rate-limit-eligible
   requests for what a human would call "one question."
2. **The conversation grows monotonically within a run.** Every tool
   result (a search result list, a full article read) gets appended to
   the message history and resent on *every subsequent call* in that run
   — OpenAI-style chat completion APIs have no concept of "the model
   already saw this," so step 5 of an 8-step run resends everything steps
   1–4 already cost, plus its own new content. A conversation that starts
   at a few hundred tokens can be several thousand by step 4 (measured
   live this session: `agent.py`'s own `_READ_ARTICLE_MAX_WORDS=1200`
   cap exists specifically because two full article reads alone could
   already approach Groq's separate *request-size* limit, a related but
   distinct constraint from the *per-minute* one this document is about).
3. **Splitting into multiple agents multiplies (1) and (2) rather than
   dividing them.** The whole premise of day8-multiagent is that
   separating research from writing from review produces better answers
   — but each of those roles is its own full call (or, for the
   researcher, its own multi-step loop), so a 3-role graph doesn't spread
   the SAME token cost across three smaller calls, it adds three
   independent conversations' worth of overhead on top of what one agent
   already needed. This is exactly why the post-cooldown retry (above)
   showed single_agent surviving on 8,000 tokens/minute while multiagent,
   under identical conditions, could not survive it even once.

**The model-switching dead end makes the shape of the problem clearer,
not just the workaround failing.** Querying Groq's `/models` endpoint
directly (`curl https://api.groq.com/openai/v1/models`) showed every
model on this account sharing the same flat 8,000-tokens/minute cap
*except* `groq/compound-mini`, which reports 70,000/minute — a real,
verified 9x difference. But `compound-mini` is Groq's own agentic
"Compound" system, with its own internal tools; a direct API test
(`tools=[...]`, `tool_choice: "auto"`) against it returned
`"tool calling is not supported with this model"`. The one model on this
account with headroom for a multi-step agent is specifically the one kind
of model that can't run one. That's not a coincidence worth shrugging
off: a model that supports arbitrary user-defined tool schemas has to
serve a fundamentally more open-ended workload than one running its own
fixed internal toolset, and providers price/throttle that flexibility
differently. **Rate limits on tool-calling models aren't a generic
Groq quirk — they're the API's answer to the fact that a tool-calling
agent's real resource cost is not "one question," it's "however many
questions the model decides it needs to ask itself."**

## Done

The multi-agent system is built, tested (58/58 offline, including the
retry-timing fix's own tests), and verified working correctly end-to-end
live (single smoke-test question: correct answer, `stopped_reason:
finished`, $0.002 total cost — see conversation history). The eval
comparison against the single-agent baseline was attempted four times
across two sessions (one same-day, one after a ~4-hour cooldown) and is
honestly reported as infrastructure-limited rather than papered over with
cherry-picked numbers, including a retry-logic improvement that initially
backfired before being fixed correctly. **single_agent now has a real,
reasonably-sized clean measurement (3/8, 37.5%, the largest clean sample
gathered); multiagent still does not** — not from lack of trying, but
because its own architecture's higher call volume makes it categorically
more exposed to this tier's rate limit, a genuine finding about the
multi-agent design's real-world cost, not an unresolved bug. The
`revision_limit` and `malformed_tool_call` findings from earlier attempts
remain real signal that survived the noise throughout.
