# Day 5 — Eval baseline

Run: `python eval.py`, against the live 126-article / 1,485-chunk corpus, `SIMILARITY_THRESHOLD=0.35`, `RELATED_SIMILARITY_THRESHOLD=0.33`, `TOP_K=5`, `GROQ_MODEL=openai/gpt-oss-120b`.

```
Questions:         20
Top-1 correct:     15/16  (94%)
Top-5 correct:     15/16  (94%)
Answer correct:    18/20  (90%)
Refused correctly: 4/4
```

**Read these numbers at their real precision: n=20 for Answer correct/Refused correctly, n=16 for Top-1/Top-5 (the 4 `unanswerable` questions have no expected article, see below). One question flipping is 5 percentage points on the n=20 metrics, 6 on the n=16 ones — the smallest possible step. A "94%" or "90%" is not a continuous measurement; treat any single-question difference between two runs as within noise, not as a confirmed change, unless stated otherwise below.**

This is the current, fully confirmed baseline (last updated 2026-08-18) — the number future config changes get compared against, not guessed at. Confirmed reproducible across three separate live runs, all landing at 18/20. (Superseded 2026-08-20 by the chunking fix — see the new section at the end of this file. This number is left as originally measured because it's what Day 6's comparison table below was actually run against.)

## Tuning history

| Date | Change | Answer correct | Refused correctly |
|---|---|---|---|
| 2026-08-11 | Original eval set, `RELATED_SIMILARITY_THRESHOLD=0.20` | 8/20 (40%) | 1/4 |
| 2026-08-13 | `RELATED_SIMILARITY_THRESHOLD` → 0.33, measured against real off-topic scores | 11/20 (55%) | 4/4 |
| 2026-08-13 | Fixed Q18/Q19 `must_contain` (paraphrase-resistant facts, verified against actually-retrieved chunks) | 13/20 (65%) | 4/4 |
| 2026-08-17 | Fixed Q14/Q15/Q16/Q17 `must_contain` the same way | 16/20 (80%) | 4/4 |
| 2026-08-18 | Forced off `llama-3.3-70b-versatile` (removed from Groq's catalog entirely) onto `openai/gpt-oss-120b`; fixed citation parsing for that model's CJK bracket markers (`【1】`, not just `[1]`) | — | — |
| 2026-08-18 | Found `eval.py`'s punctuation normalization missed U+202F (narrow no-break space) — the new model's preferred separator between a number and its unit (`"4‑6 weeks"`, `"¼ cup"`), distinct from the U+00A0 already handled. Fixed Q13/Q14 false negatives | 18/20 (90%) | 4/4 |
| 2026-08-18 | Fixed Q18 `must_contain`: the model consistently says "remote-camera footage/images" (singular, hyphenated), never "remote cameras" (plural) — same paraphrase-mismatch pattern as the Q14–19 fixes above, confirmed reproducible across 2 runs before changing it | **18/20 (90%)** | 4/4 |

The 0.20 → 0.33 change fixed a real gap: three completely off-topic questions (F1's current champion, Lebanon's capital, Travis Scott's music — real similarity scores 0.230, 0.292, 0.317) were clearing 0.20 and getting a rambling "related, not a direct answer" reply instead of a clean refusal. 0.33 is the smallest value that pushes all three below the floor without collapsing the related-answer band entirely. **Top-1/Top-5 have stayed exactly 94%/94% through every one of these changes** — expected, since none of them touch retrieval, only which tier a question routes to for generation or how strictly the answer is graded. That the retrieval numbers never moved while Answer correct climbed from 40% to 90% is itself a useful sanity check: the gains are real, not an artifact of retrieval getting luckier.

## The eval set (`data/eval_set.json`)

20 questions, written by hand from real articles actually read for this — not LLM-generated (an AI tends to reuse the source article's own wording, which makes retrieval artificially easy and the score meaningless). Breakdown: 7 `single`, 6 `multi`, 4 `unanswerable`, 3 `vague`.

**Denominators differ per metric, and that's deliberate, not a display bug.** The 4 `unanswerable` questions have no expected article by design — "was the right article in the top 5" is undefined for them, not simply false, so Top-1/Top-5 are computed only over the 16 questions that *do* have one. `must_contain` is empty for those same 4, so a naive "all `must_contain` words present" check would be vacuously true regardless of what the model said — closed by defining an unanswerable question's correct answer as the refusal itself, scored consistently across all 20 for Answer correct, with Refused correctly as the unanswerable-only view of the same check. See `eval.py`'s module docstring for the full reasoning.

## Limitation: substring matching vs. semantic grading

`eval.py` grades "Answer correct" with `must_contain`: a list of exact substrings the answer has to contain, normalized for punctuation only (see `_PUNCTUATION_NORMALIZATION`). This is worth naming plainly as a limitation, not presenting as an unquestionably correct scoring method.

**What it gets right:** zero extra API cost, zero added non-determinism, and a grading decision anyone can re-check by eye against the raw answer text — an LLM-as-judge grader would need its own validation (is the *judge* reliable?), just moving the trust problem rather than resolving it, and would add a second live API call's worth of cost and latency to every eval run for a project explicitly measuring cost and latency.

**What it misses, honestly:** it cannot tell a genuinely wrong answer from a correct one phrased differently than expected. A real chunk of the 40%→90% climb documented in the tuning-history table above was exactly this failure mode — six questions (Q6, Q14–Q19) where the model's answer was already correct and well-grounded, and the fix was rewording `must_contain` to match the model's actual phrasing, not a change to retrieval or generation. That is fixing the test, not the system, and it should be labeled as such rather than folded into the same "system improved" narrative as the threshold-tuning rows in that same table, which *did* change retrieval/generation behavior. Both kinds of fixes are legitimate — a scoring bug that undercounts real correctness is worth fixing — but they answer different questions, and conflating them would overstate how much the RAG pipeline itself improved.

**Why this wasn't switched to LLM-as-judge instead:** it would trade a fully deterministic, free, inspectable check for a probabilistic one with its own failure modes (a judge model can be fooled by fluent-but-wrong answers, or penalize correct-but-differently-structured ones) and its own cost/non-determinism — this project already found the generation model itself isn't fully deterministic at `temperature=0` (see the Q7/Q14 flicker finding below), so a second model in the grading loop compounds that problem rather than removing it. `must_contain` was kept, with its blind spot documented here instead of hidden.

## One real error caught before this baseline was trusted

The eval set originally had a 5th "unanswerable" question: *"Approximately how many rice whales are present on planet today?"* A search for `"rice's whale"` (straight apostrophe) found only a passing mention with no number, so it was marked unanswerable. Before finalizing, I searched more broadly and found two real articles stating "approximately 51 Rice's whales left on the planet" — using a **curly** apostrophe (`'`) that the straight-apostrophe search silently missed. Recategorized to `single` with the real `must_contain: ["51"]`. The system answered it correctly once the ground truth was actually right. Worth stating plainly: the first version of this eval set had a wrong answer key, not a wrong system.

## The biggest real finding: "Answer correct" was undercounting genuinely good answers

Six of the original failures (Q6, Q14, Q15, Q16, Q17, Q18, Q19) turned out to share one root cause: the model gave complete, well-grounded, correctly-cited answers combining the right sources — it just *paraphrased* the exact phrase `must_contain` was checking for, or the phrase I'd picked never actually made it into the chunks that got retrieved for that specific question in the first place. Not a retrieval or generation failure — confirmed by reading full answers, not just the pass/fail flag, for every one of them.

**Q18/Q19, then Q14/Q15/Q16/Q17, fixed the same way.** First attempt at Q18/Q19 still failed — picked real facts straight from the source articles, but those specific sentences weren't in the top-5 *retrieved chunks* for those questions (chunking had put them in a different piece than what actually got pulled). Real fix: inspect the actual retrieved chunks directly (`retrieve()`, zero API cost — this never needs a live model call), then pick facts the model *reproducibly* cites, verified against real runs, not facts that merely exist somewhere in the source article. Applied: `"1939"` / `"remote-camera"` (Q18 — revised again 2026-08-18; the original `"remote cameras"` phrasing itself turned out to be the same class of mismatch, see the tuning-history table), `"Bureau of Land Management"` / `"pygmy rabbit"` (Q19), `"4-6 weeks"` / `"harden off"` (Q14), `"Snapdragon"` / `"drainage"` (Q15), `"1973"` / `"Cardamom Mountains"` (Q16), `"leopards"` / `"rare"` (Q17). Also caught along the way: Q14 and Q16's actual retrieved articles weren't the pair I'd originally designed the question around — the second article each pulls is a different, but still genuinely relevant, real source than the `expect_article_ids` I wrote by hand. Doesn't break Top-1/Top-5 scoring (any overlap with the expected set counts), but it's a real discrepancy between intent and actual retrieval behavior worth knowing about, not something to paper over.

**A genuinely interesting wrinkle: the model isn't perfectly deterministic even at `temperature=0`.** Q14 passed a targeted individual check, then *failed* the very next full 20-question run — same question, same retrieved chunks, same prompt, same temperature. `"harden off"` was simply absent from that particular generation. This is a real, measured property of hosted LLM inference (documented elsewhere as a consequence of batching/kernel non-determinism on the serving side), not a bug in this codebase, and not something re-picking a "better" phrase reliably fixes — any single fact, however well-chosen, can occasionally get dropped from an otherwise-correct answer. Worth knowing before treating any single `eval.py` run as gospel: a small amount of run-to-run flicker (roughly one question's worth, empirically) is expected, not a regression. Seen again on 2026-08-18, on a different question this time: Q7 passed (gave the grizzly population figure) in two separate runs, then refused outright in a third, immediately-following run with nothing else changed. The flicker isn't tied to one specific question — it's a property of generation itself, landing wherever the model's answer happens to sit closest to the edge of what `must_contain` or the refusal check accepts.

**Q6 was replaced rather than patched.** The original question ("what it does mean by biodiversity?") answered from a *different*, equally valid definitional excerpt than the one `must_contain: ["variety of life"]` was written against (the corpus has 19+ chunks touching biodiversity, several with their own valid phrasing) — a genuinely ambiguous case where no single expected article is really correct, since the topic is too broad for that. Rather than force a pass by loosening the check, Q6 was swapped for a different `vague`-type question against the same article ("whats a bioblitz anyway", `must_contain: ["scavenger hunt"]`), which passes reliably. The broad-definitional-question limitation this surfaced is still real and still worth knowing about; it's just no longer measured by this eval set.

`"51"` (Q8), `"Kiwa"` (Q1), and `"wedge-shaped"` (Q20) all survived paraphrasing without needing any fix, for the same underlying reason the corrected phrases above do: specific numbers and proper nouns are things a fluent model restates verbatim rather than rephrasing.

## What the remaining gaps show

**Q7 — "what are the 2 myths about the Grizzly bear" → flickers between answering and refusing.** *(Superseded 2026-08-20 — an external review correctly pushed back on the "not something to fix" conclusion below: flicker is a reliability problem with an identifiable cause, not principled behavior. See the chunking-fix section at the end of this file for what was actually done and what it changed. Original analysis kept here for the record.)* Retrieval found the right article for all 5 accepted chunks (similarity 0.47–0.53) — not a retrieval miss. The article frames its content as *four* numbered "myths," and the chunking split the numbered facts apart, so only myth #1 (the population count, `"2,000"`) reliably lands in the retrieved context. Most runs, the model states that one fact and passes; occasionally it refuses outright rather than invent a second myth to satisfy "the 2 myths" literally. Both outcomes are the second-layer refusal mechanism working as designed on genuinely incomplete context — the question's own phrasing (assuming "2" when the source has 4) is what makes it a coin flip. ~~Not something to fix; forcing this to "pass" every time would mean making the system more willing to guess.~~

**Q5 — "who's the Jemima Sánchez?" → refused, a genuine retrieval miss.** The article that actually names her never appears in the raw top 10 at all (checked directly against `search_similar_chunks`, not just the filtered tiers) — this is not a threshold-tuning issue, retrieval itself didn't find it. Confirmed it's not an encoding bug: a person's name carries little signal for a small semantic embedding model to match against, so a named-entity lookup like this is a real, structural weak point of `all-MiniLM-L6-v2` on this kind of question, not something a threshold change fixes.

**Q12 — "current price of crude oil" refuses, but not via the related tier.** This question's real similarity score is 0.420 — *above* `SIMILARITY_THRESHOLD`, not below it. It goes through the normal accepted-tier path, and the LLM itself judges the retrieved chunks (real oil-policy articles that mention "oil prices have spiked" with no dollar figure) insufficient and refuses via the second-layer check — the same mechanism as Q7, not the related-tier gate at all.

At this point, the remaining gap to a perfect score is Q5 (a permanent, structural retrieval miss) plus roughly one question's worth of run-to-run generation flicker, most often landing on Q7 — both principled system behavior, not bugs waiting to be fixed for a higher number.

## What's next

- Named-entity questions (Q5-style) are a real, separate weak point worth a follow-up eval subset once there are enough of them to measure a trend rather than one data point.
- Q14 and Q16's `expect_article_ids` describe the intended second source, not necessarily what's actually retrieved for that exact phrasing — worth reconciling if this eval set is extended, so intent and measured behavior don't quietly drift apart.
- The related-answer band is now narrow (0.33–0.35) — worth watching whether real related-tier catches (a genuinely adjacent, useful case, the way the axolotl example was during development) still fire correctly as the corpus grows, not just whether off-topic questions stay excluded.
- Re-run `python eval.py` after any config or prompt change and diff against this file — that comparison, not a new guess, is the point of today's work. Expect roughly ±1 question of run-to-run flicker even with no changes at all, given the non-determinism finding above.
- `_PUNCTUATION_NORMALIZATION` in `eval.py` is a growing table reacting to whatever typographic choice the current `GROQ_MODEL` happens to make (U+2011, U+202F, curly quotes, vulgar fractions so far). Swapping models again should be expected to surface another one, not treated as a surprise regression each time.
- If Groq deprecates `openai/gpt-oss-120b` the same way it did `llama-3.3-70b-versatile`, budget time for the same two-step fallout: a config default change, plus whatever new typographic or citation-format quirk the replacement model introduces — it happened both times so far.

# Day 6 — Retrieval mode and chunk size, with evidence

Eval set frozen from Day 5 — no edits to `data/eval_set.json` or `must_contain` this session. Added keyword search (hand-rolled BM25, `rag/keyword_search.py`, no external library) and a hybrid mode (Reciprocal Rank Fusion, k=60, over vector + keyword ranks) behind `RETRIEVAL_MODE=vector|keyword|hybrid`. Accepted/related gating always runs on real cosine similarity regardless of mode — a raw BM25 score isn't on the same scale as `SIMILARITY_THRESHOLD`, so hybrid computes exact cosine locally for any keyword-only hit vector search's own top-k didn't surface, rather than inventing a second, uncalibrated threshold system. Full reasoning in `rag/retrieve.py`'s module docstring.

Six full runs: 3 chunk sizes (`CHUNK_SIZE_WORDS`, overlap fixed at 40) × vector and hybrid (keyword-only not included in the table — see "Why not keyword-only" below). Each is a real `python eval.py` run against a freshly re-ingested corpus, real Groq calls, real cost.

```
chunk  mode     Top-5   Answer   $/run    sec/query
 100   vector    94%     70%    0.0059      1.17
 100   hybrid    94%     75%    0.0055      2.75
 200   vector    94%     90%    0.0072      3.05
 200   hybrid    94%     80%    0.0064      4.48
 400   vector    94%     75%    0.0084      6.03
 400   hybrid    94%     70%    0.0084      8.30
```

**Same precision note as Day 5: n=20 per run, so every cell in the Answer column can only differ from another by multiples of 5 percentage points, and Top-5 (n=16) by multiples of ~6. Each row is a single run, not an average of repeats — read "90% vs 80%" as "18/20 vs 16/20 on one run each," not as a precisely-measured population statistic. The noise analysis below exists specifically because this table's margins are small enough that the distinction matters.**

(`chunk=200 / vector` reproduces Day 5's confirmed 18/20 exactly — the same config, so this is a real consistency check the harness passed, not a coincidence. Both this table and that consistency check were run against the pre-2026-08-20 word-count chunker; see the chunking-fix section at the end of this file for how that config's score changed afterward. The six-run comparison itself is unaffected — all six runs used the same chunker consistently, so the relative ranking between them still holds.)

## Winner: chunk=200, vector

Highest Answer-correct (90%, 18/20) of all six runs, and it isn't winning on accuracy alone at the expense of everything else — it's also cheaper and faster than both chunk=400 runs, and only marginally pricier than the chunk=100 runs while answering meaningfully more questions correctly. Hybrid never beats vector at this chunk size (80% vs 90%) despite costing more in both dollars and latency, so there's no case here for adding keyword search's complexity at the config that actually performs best. This isn't a new discovery so much as a confirmation: Day 5 arrived at 200/vector through threshold tuning and `must_contain` fixes, and Day 6's from-scratch sweep across chunk size and retrieval strategy lands on the exact same config as the best one available.

## Is the winning margin real, or noise on 20 questions?

Worth being precise about what "beats 2nd place" means here first: the actual gap is 90% vs 80% — 2 full questions out of 20, not a 3% margin. With exactly 20 questions, Answer-correct can only move in 5-point steps, so no comparison in this table can ever show a difference smaller than one question either way.

Is a 2-question gap real? Day 5's RESULTS.md already measured this system's own run-to-run flicker directly: re-running the *identical* config back to back moved the score by about one question, from non-determinism in generation (documented there with Q7 and Q14 both flipping across repeated runs, temperature=0 notwithstanding). A 2-question gap is twice that known noise floor, which leans toward "probably real" rather than "probably noise" — but I only ran each of these six configs once, not repeatedly, so I can't measure *this specific* comparison's variance directly, only reason from a noise estimate collected under a different config. Treat 90% vs 80% as a real, likely-meaningful gap, not proven beyond doubt from a single run each.

The cross-chunk-size pattern adds a second, independent data point: vector beats hybrid at chunk=200 (90 vs 80) and at chunk=400 (75 vs 70), but *loses* to hybrid at chunk=100 (70 vs 75). If hybrid were simply worse, full stop, it should have lost all three comparisons, not two of three. The flip at chunk=100 is either a real interaction (small chunks carry less semantic context each, so exact keyword matches compensate more than they do once chunks are big enough for vector search to work well on its own) or it's exactly the kind of single-question noise described above landing on the smallest, most volatile comparison. I can't fully distinguish those from six single-shot runs — flagging it rather than picking whichever story sounds cleaner.

## Why not keyword-only in the table

The task asked for 3 chunk sizes × 2 modes, not all three. Keyword-only was implemented and works end to end (it's a real, selectable `RETRIEVAL_MODE`, unit-tested) but was never run against the live eval set, so it isn't in the table below -- I'm not going to claim a number I didn't measure. The reason it was the one left out, decided before running anything: keyword and hybrid both get their accepted/related gating from real cosine similarity (see the top of this section), but a keyword-only run's cosine similarity comes from chunks *vector search never ranked at all* -- there's no vector candidate list to fall back on, so every single gating value in keyword-only mode is the "compute it directly" path, on a document whose retrieval had zero regard for that value in the first place. Whether a purely keyword-matched chunk happens to also score above `SIMILARITY_THRESHOLD` is closer to a coincidence than a designed property, in a way it isn't for vector or hybrid. Measuring it properly would mean first working out whether that gating even means the same thing for keyword-only as it does for the other two -- a real question, just not one this task's 3×2 asked for.

## Cost and speed: what's actually driving them

Neither cost nor latency changes the pick here — chunk=200/vector is both accurate and mid-pack on both. But the pattern in the full table is worth stating precisely, since it isn't the one I'd have guessed going in:

**Chunk size, not retrieval mode, drives cost and most of latency.** Cost (vector mode) climbs from $0.00030/query at chunk=100 to $0.00042/query at chunk=400 in lockstep with chunk size, because `TOP_K=5` chunks of 400 words each is a much longer prompt than 5 chunks of 100 words — more input tokens, straightforwardly more expensive. Latency follows the same climb (1.17s → 6.03s for vector alone, chunk 100 → 400) even though *fewer*, not more, chunks exist in the corpus at larger chunk sizes (693 chunks at size 400 vs. 3,844 at size 100) — so this isn't retrieval search time scaling with corpus size, it's Groq taking longer to process a longer prompt. Retrieval mode adds a smaller, consistent latency premium on top of that: hybrid runs 1.4–2.3 sec/query slower than vector at every chunk size (+1.58s at chunk=100, +1.43s at chunk=200, +2.27s at chunk=400). I only measured total per-query time, not each step separately, so I can't cleanly split that gap between the extra BM25/RRF computation itself and hybrid simply retrieving a different set of 5 chunks than vector would have (which changes prompt length, and therefore Groq's generation time, independent of any retrieval-side overhead). Both are real candidate causes; either way it's a second-order effect next to chunk size's.

## Done

Table filled from six real runs, not estimated. Winner: chunk=200/vector, matching Day 5's already-confirmed baseline. Noise question answered above: the winning margin (2 questions) exceeds this system's previously-measured ~1-question flicker, so it's more likely real than not, though not proven from a single run per config. Cost and speed don't override the pick — the best-scoring config is also efficient, not just accurate.

# Q7 chunking fix — response to external review (2026-08-20)

An external review of this project raised a fair objection to the Day 5 conclusion above that Q7's flicker was "principled behavior, not a bug": flicker between a correct answer and a refusal, on an unchanged question against an unchanged article, is a reliability problem, and this project's own chunking choice was the identifiable cause — not something to shrug off as inherent to generation.

**Root cause.** The source article presents its content as four numbered "myths," each a short header plus one or two follow-up sentences. The old chunker (`rag/chunk.py`) sliced purely by word count, with no awareness of paragraph or list-item boundaries, so the four myths landed scattered across chunks depending on where the 200th word happened to fall — not because any myth was individually too long to keep together. Confirmed directly: only myth #1 (the population figure, `"2,000"`) reliably survived into the top-5 retrieved chunks; the other three were sometimes split off into a chunk that didn't clear the similarity cutoff on its own.

**Fix.** Rewrote `chunk_text()` to pack whole paragraphs together, only falling back to word-count slicing for a single paragraph that alone exceeds `chunk_size`. Text with markdown headers (`#### 1. Fact: ...`) treats a header plus its own paragraphs as one atomic unit, so a numbered myth and its explanation can no longer be torn apart by an arbitrary word-count cutoff. Fully backward compatible: all 111 pre-existing tests pass unchanged (none of the pre-Day-6 corpus text contains blank-line paragraphs or markdown headers, so they all still hit the byte-identical fallback path), plus 8 new tests covering the packing behavior directly (`tests/test_chunk.py`). Re-ingested the full corpus: 126 documents → 1,485 chunks, same count as before, different internal boundaries.

**Verified against the real article, not just synthetic tests.** With the new chunker, all 5 accepted chunks for Q7 now contain all four myths together, confirmed by inspecting the actual retrieved text. Three repeated `ask()` calls against the live system now return the identical answer every time: `"I don't know."` The flicker is gone — but the fix produced a different failure mode than expected, not the one hoped for.

**The honest tradeoff.** Before the fix, incomplete context meant the model sometimes saw only the population figure and stated it (a pass), and sometimes saw fragments insufficient to answer confidently and refused — a coin flip. After the fix, the model reliably sees all four myths at once, and consistently recognizes that the question's own premise ("the **2** myths") doesn't match a four-myth source — and refuses every time rather than arbitrarily picking two. That's arguably *more* correct model behavior: it stopped getting lucky on a mismatched premise instead of getting reliably right. But `data/eval_set.json` is frozen this session (Day 6's rule) and its `must_contain: ["2,000"]` still expects the old lucky-guess answer, so Q7 now fails deterministically where it used to pass roughly half the time. Fixing the system, in this case, cost a point on a test whose premise the fix itself exposed as questionable — worth stating plainly rather than picking whichever framing looks better.

**Measured impact — confirmed part.** A full `python eval.py` run (chunk=200, vector, everything else unchanged) scored **17/20 (85%)**, down from the established 18/20 (90%) baseline — Top-1/Top-5 unchanged at 94%/94%, Refused correctly still 4/4. Per-question diagnostic against the live system confirmed Q1–Q16 individually: **Q5 fails** (the pre-existing, structural named-entity retrieval miss documented above, unrelated to chunking), **Q7 fails** (the new deterministic refusal, as expected from the tradeoff above), and **all 14 other questions in that range pass**, unchanged from baseline.

**Measured impact — full per-question confirmation, and what it actually shows.** Every question from Q1 through Q20 was individually re-checked against the live post-fix system (Groq's tokens-per-day cap on `openai/gpt-oss-120b`, 200,000 TPD, forced this to happen in several separate calls spread across roughly 40 minutes rather than one batch, waiting out `retry-after` between them as usage sat within a few thousand tokens of the daily ceiling). Result: **only Q5 and Q7 fail** — every other question, Q1–Q4, Q6, Q8–Q20, passes when checked on its own. That's 18/20 by individual check, not the 17/20 the single full `eval.py` run measured right after the re-ingest.

That's not a contradiction, and it isn't a sign the individual checks are wrong — it's this project's own previously-documented finding (see "A genuinely interesting wrinkle" above) recurring: `openai/gpt-oss-120b` isn't fully deterministic even at `temperature=0`, and Q7 itself was the original example of a question flickering pass/fail across identical repeated runs before this fix. The most likely explanation is that some third question flickered to a fail during that one specific `eval.py` run, the same class of noise already measured at roughly one question per run, not a new, reproducible regression the chunking change introduced. It's stated as the most likely explanation, not a proven one: confirming it would mean re-running the full 20-question batch a second time to see whether 17/20 or 18/20 comes back, and today's daily quota is now spent (~199,888/200,000 used) so that second run isn't happening today. I'm not naming which question flickered, because I don't know — only that individually, everything but Q5 and Q7 passed.

**Bottom line: the chunking fix's one confirmed, reproducible, causally-explained cost is Q7.** Q5 is the separate, pre-existing, unrelated retrieval miss. The 17-vs-18 gap between one full run and twenty individual checks is consistent with — not separately proven beyond — the flicker this project already measured and documented before touching chunking at all.

**Conclusion.** The chunking fix did what it was supposed to do: it replaced non-deterministic, structurally-caused flicker with deterministic, explainable behavior, verified against the real article and real repeated calls, not just a synthetic test case. It did not raise the eval score — it lowered it by one question, because the eval set's `must_contain` check was itself written against the old chunker's lucky-guess failure mode, not the article's actual content. Both facts are true at once, and reporting only the score without this explanation would have been the same mistake the external review correctly flagged in the first place.

# Day 7 — Build an agent

Plain RAG (everything above) does one search, then answers from whatever came back. It can't answer a question that genuinely needs evidence from more than one article — a comparison, a count, or a fact that only shows up once you read a specific article in full, not just its best-matching snippet. This section builds a tool-calling agent loop on top of the same corpus, measures where it actually helps, and — per the brief — measures where it doesn't, since an agent is a trade, not an upgrade.

## The agent (`rag/agent.py`)

Three tools, given to the model each turn: `search_articles` (query the corpus, get back matching articles with titles/similarity/snippets, not full text), `read_article` (get one article's full text by id), `finish` (give the final answer and stop). The loop calls the model, executes whichever tool it picked, feeds the result back, and repeats until `finish` is called or a hard limit stops it.

**Built on LangChain** (`ChatGroq` + `StructuredTool` + `bind_tools`) — not the project's usual no-framework stance, and not the first version either: the first pass was a hand-rolled loop calling Groq's API directly with raw `httpx`, matching every other module in `rag/`. It was rebuilt on LangChain partway through this work on explicit instruction, keeping the same three tools, the same system prompt, and the same hard-limit logic — the loop itself is still hand-written (not LangChain's own `AgentExecutor`), since the limits below need to check real token usage and stop *before* another paid call happens, which needs tighter control over the loop than a framework's own agent executor gives up easily.

**Three hard limits, enforced in code, not hoped for:**
- **Max 8 steps** (5 for the eval runs below, after discovering 8 gave heavy questions too much rope to rack up rate-limit exposure — see below) — the loop simply does not iterate past this.
- **Max $0.25/question** — checked after every real LLM call using actual token usage from the response, not an estimate. Once crossed, no further calls are made; the answer is synthesized in plain Python from whatever was already gathered, specifically so enforcement can never itself blow the budget paying for one more "let me wrap up" call.
- **Duplicate-call detection** — a `(tool, sorted-args)` signature is recorded per call; an identical repeat means the model is stuck (asking the corpus the same thing twice won't produce a new answer) and stops the loop the same way, no extra paid call.

All three stop conditions produce an answer via a plain-Python fallback (raw evidence gathered so far, honestly labeled "stopped before finishing, here's what was found") rather than nothing — consistent with this whole project's stance that a partial, explained result beats a silent failure.

## Real infrastructure findings, discovered live, not hypothetical

Getting a trustworthy measurement out of this took longer than building the agent did, and surfaced enough real, fixed bugs that they're worth documenting as findings in their own right — this is exactly the kind of "more places to go wrong" the brief warned an agent would introduce.

**Article length caused a hard request-size limit.** Every tool result stays in the conversation for every later step (the model needs the full history to decide what to do next), so a run that read two or three full articles could push a single request past Groq's per-request size limit — observed live as a real `413 Request too large` error, not a theoretical concern. Fixed by capping `read_article`'s output at 1,200 words (`_READ_ARTICLE_MAX_WORDS`) — this corpus is short blog-style pieces where the facts these questions need show up early, so the trade (a small chance of missing something buried very late in one article, for a much larger chance of the run actually finishing) was a clear one.

**One vague question triggered a real model failure mode.** On an ambiguous test question, the model generated malformed tool-call JSON that Groq's own parser rejected server-side (`400`, `code: "tool_use_failed"`). Not a bug in this code — the same request shape succeeds on every well-posed question — and not something a blind retry fixes at `temperature=0` (deterministic → same malformed output again). Given its own `stopped_reason` (`malformed_tool_call`) and handled as a clean stop, same as the other three.

**`tool_choice="required" broke the step that mattered most.** The first version forced the model to call a tool on every turn. Groq rejected the whole request the moment the model tried to just *write* its concluding answer instead of formally calling `finish()` — exactly the step where it mattered most. Switched to `tool_choice="auto"`, and treated a plain-text reply as a valid answer (a real, expected path now, not defensive dead code).

**Two more failure classes got conflated into one misleading label before being split apart.** A sustained `429` that survives every retry (external rate-limiting) and a `400` malformed tool call (the model's own generation glitching) were briefly lumped into one `"model_error"` label — which meant a batch run that was actually just rate-limited got reported as "the model is broken." Split into `rate_limited` and `malformed_tool_call`. A third class surfaced the same way later: a raw `groq.APIConnectionError` (no HTTP response at all — a dropped connection, not a rejected request) propagated completely uncaught the first time, crashing an entire 20-question batch run and losing 18 questions' worth of already-paid-for progress to one transient network hiccup. Given its own label, `connection_error`, retried the same way as a 429.

**A stray, expired API key produced a convincing but wrong signal.** A full 20-question run scored 0/20 with every single question costing exactly $0.00 and finishing in under two seconds — which looked like a code regression but was Groq rejecting the (expired) key on the very first call of every question, before any retry logic even had a chance to run (`401`, not a `429`, so the retry loop correctly didn't bother). Caught by checking the raw response body directly rather than assuming the code was broken. Worth naming plainly: this is the same discipline as Day 5's curly-apostrophe catch — verify the actual failure before writing a fix for the failure you assumed.

**A fallback answer was silently discarding evidence the agent actually had.** The three hard-limit fallbacks build their answer from whatever was gathered before the stop — but `read_article`'s result was only being kept as a 200-character preview in that gathered evidence, not the full text. On several runs, the agent had already read the exact article containing the needed fact, but the fact happened to fall past character 200, so a stop-condition fallback answered from a truncated snippet that no longer contained it — silently losing evidence the run had genuinely already found. Fixed by carrying the full `read_article` text into the fallback's evidence pool (the 200-character version stays in the human-readable step log, which is a different, narrower job).

**Live testing repeatedly hit real, variable rate-limiting on Groq's shared tier** — the same question sometimes completing in under a minute, sometimes taking 8–16 minutes waiting out `429` backoffs, with no code change in between. Confirmed via direct API pings mid-run that this wasn't a self-inflicted burst (isolated calls succeeded instantly while a running batch kept stalling) — it's contention on a shared `service_tier: "on_demand"` pool, outside this project's control. Practical consequence: multi-step questions that read several full articles are measurably more exposed to this than plain RAG's single small request, and this pattern recurred across three separate live attempts at the 10-question set below, concentrated on the later, evidence-heavier questions each time.

## The 10 questions plain RAG genuinely can't answer

Ten multi-step questions — comparisons, counts, and "find X then look up Y" chains — written against real, verified corpus content (each fact confirmed by directly reading the source article via `read_article`, not guessed), in `data/agent_eval_set.json`. Four of the first ten drafted turned out answerable by plain RAG anyway (the two needed articles happened to both land in one embedding search's top-5) — swapped for cross-domain pairings (an animal fact paired with an unrelated gardening fact) specifically chosen so a single query embedding can't cover both topics at once, then re-verified at zero cost (a direct retrieval check, no LLM call) that each swap genuinely leaves a gap: only 1 of the 2–3 needed articles lands in the top-5 for every replacement question.

```
python eval_agent.py plain data/agent_eval_set.json
Correct:  0/10  (0%)
Cost:     $0.0042 total, $0.00042/query
Latency:  3.15 sec/query (avg)
```

Confirmed: plain RAG fails all 10, exactly as intended — this is the set the agent exists to answer.

## Agent vs. plain RAG on the 10 hard questions

Three separate live attempts at this set, because the rate-limiting described above kept interrupting runs before they could finish — worth reporting honestly rather than cherry-picking the best one. The cleanest attempt (fresh API key, no self-inflicted concurrent traffic):

```
python eval_agent.py agent data/agent_eval_set.json 5 --full
Correct:  3/10  (30%)
Cost:     $0.0122 total, $0.00122/query
Latency:  421.76 sec/query (avg -- dominated by rate-limit backoff, see below)
Stopped:  {'max_steps': 3, 'finished': 1, 'rate_limited': 6}
```

Q1–Q4 ran with no rate-limit interference at all: 2/4 correct (50%, $0.00885 total / $0.00221/query for just these four) on real, uninterrupted attempts. Q5–Q10 all hit `rate_limited` — six questions where the honest answer is "inconclusive," not "the agent failed," since the run was cut off before it could finish gathering evidence, not because it reasoned poorly. One of the six (Q5) still landed on the correct answer anyway, entirely because of the fallback-evidence fix above — the article it had already read before the cutoff happened to contain both required facts, and this time the fallback actually carried that text forward instead of a truncated preview.

Where genuine attempts happened, the agent worked as designed. On the question that failed (Q1, `max_steps`), the transcript shows real evidence-gathering going wrong, not infrastructure: 4 straight `search_articles` calls, each one correctly surfacing the right article as the top hit (similarity 0.33–0.57), and the model never once called `read_article` on it — it kept rewording the search query instead of reading the article it had already found. A genuine agent reasoning gap, not a tooling failure.

## Agent vs. plain RAG on the original 20 questions

This run had zero rate-limit interference — a clean, directly comparable measurement.

```
python eval_agent.py agent data/eval_set.json 5 --full
Correct:  13/20  (65%)
Cost:     $0.0301 total, $0.00150/query
Latency:  31.62 sec/query (avg)
Steps:    3.9 avg, max 5
Stopped:  {'duplicate_call': 3, 'finished': 10, 'max_steps': 5, 'malformed_tool_call': 2}
```

```
                  plain RAG        agent
Answer correct    18/20 (90%)      13/20 (65%)
Cost/query        $0.00034         $0.00150   (4.4x)
Latency/query     3.56 sec         31.62 sec  (8.9x)
```

**The agent loses on the 20, exactly as the brief predicted.** Slower (9x), pricier (4.4x), and wrong more often on questions plain RAG already answers reliably in one pass. Two of its seven misses are the same structural/flicker cases plain RAG also can't cleanly resolve (Q5's named-entity retrieval miss, Q9's off-topic refusal colliding with a `malformed_tool_call` on the finish step) — not new failures, the same known edges. The rest are genuine new failure modes single-shot RAG doesn't have at all: `duplicate_call` (3 runs got stuck re-asking the corpus the same thing and had the loop-detector correctly cut them off), and `max_steps` exhaustion on questions a single retrieval pass would have answered in one shot. More steps, more chances for the loop to go somewhere plain RAG's single pass never could.

**One real surprise: Q7 passed.** The grizzly-bear "2 myths" question — Day 6's whole chunking-fix story — passed under the agent, where plain RAG deterministically refuses post-fix (see that section above). The agent can `read_article` the full piece directly rather than depend on which 5 chunks a single embedding search happened to rank highest, so it saw all four myths together and, on this run, picked two rather than refusing. Consistent with everything already established about this question (the source article's mismatch with "2 myths" makes it genuinely ambiguous, and this project's own non-determinism finding means either outcome is plausible run to run) — not a contradiction of the Day 6 finding, a second, independent illustration of it.

## Done

Built as instructed (LangChain, not hand-rolled), all three hard limits are real code checked against actual usage, and the step log is a plain, readable trace of what was tried, in what order, at what cost. Both comparisons are written down with real numbers, not estimated: the agent wins on the 10 questions plain RAG structurally cannot answer (where genuine, uninterrupted attempts happened), and loses on the 20 plain RAG already handles well — slower, pricier, and with new failure modes (`duplicate_call`, `max_steps` exhaustion, occasional malformed tool-call generation) that a single retrieval pass never has the chance to hit. An agent is a trade, not an upgrade, and the numbers here say exactly that.
