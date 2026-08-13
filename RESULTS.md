# Day 5 — Eval baseline

Run: `python eval.py`, against the live 126-article / 1,485-chunk corpus, `SIMILARITY_THRESHOLD=0.35`, `RELATED_SIMILARITY_THRESHOLD=0.20`, `TOP_K=5`.

```
Questions:         20
Top-1 correct:     15/16  (94%)
Top-5 correct:     15/16  (94%)
Answer correct:    8/20  (40%)
Refused correctly: 1/4
```

This is today's baseline (2026-08-11) — the number future config changes get compared against, not guessed at.

## The eval set (`data/eval_set.json`)

20 questions, written by hand from real articles actually read for this — not LLM-generated (an AI tends to reuse the source article's own wording, which makes retrieval artificially easy and the score meaningless). Breakdown: 7 `single`, 6 `multi`, 4 `unanswerable`, 3 `vague`.

**Denominators differ per metric, and that's deliberate, not a display bug.** The 4 `unanswerable` questions have no expected article by design — "was the right article in the top 5" is undefined for them, not simply false, so Top-1/Top-5 are computed only over the 16 questions that *do* have one. `must_contain` is empty for those same 4, so a naive "all `must_contain` words present" check would be vacuously true regardless of what the model said — closed by defining an unanswerable question's correct answer as the refusal itself, scored consistently across all 20 for Answer correct, with Refused correctly as the unanswerable-only view of the same check. See `eval.py`'s module docstring for the full reasoning.

## One real error caught before this baseline was trusted

The eval set originally had a 5th "unanswerable" question: *"Approximately how many rice whales are present on planet today?"* A search for `"rice's whale"` (straight apostrophe) found only a passing mention with no number, so it was marked unanswerable. Before finalizing, I searched more broadly and found two real articles stating "approximately 51 Rice's whales left on the planet" — using a **curly** apostrophe (`'`) that the straight-apostrophe search silently missed. Recategorized to `single` with the real `must_contain: ["51"]`. The system answered it correctly once the ground truth was actually right. Worth stating plainly: the first version of this eval set had a wrong answer key, not a wrong system.

## The biggest real finding: "Answer correct" undercounts genuinely good answers

**8/20 (40%) understates how well the system actually did.** Digging into the specific failures (Q6, Q18, Q19 — 3 of the 12 content-bearing questions) shows a consistent pattern: the model gave complete, well-grounded, correctly-cited answers that combined the right sources — it just *paraphrased* the exact phrase `must_contain` was checking for.

- **Q18** ("How did wolves in the Northwest go extinct and how are they tracked today?") — real answer covers both articles correctly (extinction history + today's winter-count tracking method), but says "human activities" where the article said "trapping," and describes the tracking method in its own words instead of saying "breeding pairs" verbatim.
- **Q19** ("What is being done to help the sage grouse and what other animals depend on sagebrush?") — same story: says "federal land use plans" instead of "Resource Management Plans," though it did happen to use "pygmy rabbit" verbatim from the other article.
- **Q6** (biodiversity definition) — answered from a *different*, equally valid definitional excerpt than the one `must_contain: ["variety of life"]` was written against (the corpus has 19+ chunks touching biodiversity, several with their own valid phrasing).

This is a strict literal-substring-match methodology issue, not a retrieval or generation failure — confirmed by reading the full answers, not just the pass/fail flag. **Next eval set should prefer `must_contain` phrases that are hard to paraphrase** — specific numbers, proper nouns, named quantities — over descriptive phrases a fluent model will naturally restate in its own words. `"51"` (Q8), `"Kiwa"` (Q1), `"wedge-shaped"` (Q20), and `"429"` (Q17) all survived paraphrasing for exactly this reason; generic phrases like `"Resource Management Plans"` didn't.

## What the other failures show

**Q7 — "what are the 2 myths about the Grizzly bear" → refused, correctly.** Retrieval found the right article for all 5 accepted chunks (similarity 0.47–0.53) — not a retrieval miss. But the article frames its content as *four* numbered "myths," and the chunking split the numbered facts apart: only myth #1 (the population count) landed cleanly in the retrieved context. Rather than invent a second myth to satisfy "the 2 myths" literally, the model refused. That's the second-layer refusal working as designed, at a real cost to recall — and a legitimate case where the question's own phrasing (assuming "2" when the source has 4) contributed to the miss.

**Q5 — "who's the Jemima Sánchez?" → refused, a genuine retrieval miss.** The article that actually names her never appears in the raw top 10 at all (checked directly against `search_similar_chunks`, not just the filtered tiers) — this is not a threshold-tuning issue, retrieval itself didn't find it. Confirmed it's not an encoding bug: a person's name carries little signal for a small semantic embedding model to match against, so a named-entity lookup like this is a real, structural weak point of `all-MiniLM-L6-v2` on this kind of question, not something a threshold change fixes.

**Q9, Q10, Q11 — off-topic questions (F1, Lebanon, Travis Scott) got a rambling "related, not a direct answer" instead of a clean refusal.** This is the most actionable finding from this baseline. The related-answer tier (added this week, `RELATED_SIMILARITY_THRESHOLD=0.20`) is firing even for topics that share essentially nothing with the corpus — something *always* clears 0.20 out of 1,485 chunks, so the floor isn't actually filtering out "nothing is even close," just "nothing is very close." `RELATED_SIMILARITY_THRESHOLD` is a real candidate to raise next — worth an `eval.py`-driven sweep the same way `SIMILARITY_THRESHOLD` itself was tuned, rather than guessing at a new value.

**Q12 — "current price of crude oil" refused cleanly** despite the corpus containing real oil-related content (energy policy articles mention "oil prices have spiked," with no dollar figure anywhere) — worth noting only because it shows the related-tier miss above isn't universal; it didn't fire here even though there's real oil-adjacent vocabulary in the corpus, which cuts against a simple "just raise the floor" fix being the whole story.

## What's next

- Rewrite `must_contain` for Q6, Q18, Q19 to use paraphrase-resistant facts (specific numbers/names) instead of descriptive phrases, or accept multiple valid phrasings per question — the current failures there are a grading-methodology gap, not a system regression.
- Sweep `RELATED_SIMILARITY_THRESHOLD` against this eval set (same method the README's `SIMILARITY_THRESHOLD` sweep already used) before just guessing a higher number.
- Named-entity questions (Q5-style) are a real, separate weak point worth a follow-up eval subset once there are enough of them to measure a trend rather than one data point.
- Re-run `python eval.py` after any config or prompt change and diff against this file — that comparison, not a new guess, is the point of today's work.
