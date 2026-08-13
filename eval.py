"""Day 5 -- measures this RAG system against a fixed, hand-written question
set (data/eval_set.json), so a config change later this week can be
compared against today's numbers instead of guessed at.

Run: python eval.py

Every question in the eval set was written by hand from real articles in
the corpus (see data/eval_set.json's own questions) -- an LLM-generated
question tends to reuse the source article's exact wording, which makes
retrieval artificially easy and the score meaningless.

DENOMINATORS, AND WHY THEY DIFFER PER METRIC:

Not every question has a "correct article" to find -- the 4 unanswerable
ones (data/eval_set.json, type="unanswerable") have an empty
expect_article_ids by design, so "was the right article in the top 5" is
undefined for them, not simply false. Top-1/Top-5 are therefore computed
only over the 16 questions that DO have an expected article -- forcing
them into a false/true bucket for an article that was never supposed to
exist would just be noise dressed up as a number.

Answer correct and Refused correctly are the reverse split: for the 4
unanswerable questions, must_contain is empty, so a naive "every
must_contain word is present" check is vacuously true regardless of what
the model actually said -- that loophole would let the related-answer
tier or a hallucination score as "correct" for free. So an unanswerable
question's "correct answer" is defined as the refusal itself, and Answer
correct is scored consistently across all 20 questions on that basis.
Refused correctly is the narrower, unanswerable-only view of the same
check, matching the brief's own split between the two metrics.
"""

import json
import time

import httpx

from rag.ask import _is_refusal, ask
from rag.db import search_similar_chunks
from rag.embed import embed_query

# 17 sequential Groq calls can trip the API's per-minute token budget even
# though no single call is doing anything wrong (measured: 12,000
# tokens/minute on this key, refilling in well under a minute) --
# generate.py doesn't retry on its own (a 429 there propagates as a raw
# httpx.HTTPStatusError), so without this a transient throttle would crash
# the whole eval run instead of just costing a few seconds. Exponential,
# not fixed, because a single 15s wait sometimes wasn't enough on a
# thoroughly-warmed-up key during testing.
_RATE_LIMIT_RETRIES = 4
_RATE_LIMIT_BASE_BACKOFF_SECONDS = 15

# Between every question, not just after a failure -- see run_eval().
_INTER_QUESTION_DELAY_SECONDS = 5


def _ask_with_retry(question: str) -> dict:
    for attempt in range(_RATE_LIMIT_RETRIES + 1):
        try:
            return ask(question)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 429 or attempt == _RATE_LIMIT_RETRIES:
                raise
            time.sleep(_RATE_LIMIT_BASE_BACKOFF_SECONDS * (2**attempt))
    raise AssertionError("unreachable")

EVAL_SET_PATH = "data/eval_set.json"

# How many raw, UNFILTERED chunks to pull per question when checking
# Top-1/Top-5. Deliberately not going through retrieve()'s
# SIMILARITY_THRESHOLD/RELATED_SIMILARITY_THRESHOLD gates -- Top-1/Top-5
# measure the vector search's raw ranking (did it surface the right
# article at all), a different question from "was it good enough to
# answer from." Conflating the two would hide a real retrieval miss
# behind the refusal gate correctly kicking in for an unrelated reason.
_RAW_FETCH_K = 15


def _top_ranked_articles(question: str, limit: int = 5) -> list[str]:
    """The first `limit` DISTINCT article URLs among the raw top-K chunks,
    in rank order. Several chunks from the same densely-chunked article
    must not each consume a separate "slot" -- that would make Top-5 an
    easier bar to clear for a long article than a short one.
    """
    chunks = search_similar_chunks(embed_query(question), _RAW_FETCH_K)
    seen: list[str] = []
    for c in chunks:
        url = c["document_url"]
        if url not in seen:
            seen.append(url)
        if len(seen) >= limit:
            break
    return seen


def _answer_is_correct(entry: dict, answer: str) -> bool:
    """See the module docstring for why unanswerable questions are scored
    against refusal rather than must_contain (which is empty for them)."""
    if entry["type"] == "unanswerable":
        return _is_refusal(answer)
    return all(phrase.lower() in answer.lower() for phrase in entry["must_contain"])


def run_eval() -> None:
    with open(EVAL_SET_PATH, encoding="utf-8") as f:
        eval_set = json.load(f)

    top1_hits = top5_hits = answer_hits = refusal_hits = 0
    retrieval_graded = unanswerable_total = 0

    for i, entry in enumerate(eval_set):
        if i > 0:
            # Paced, not just reactive: firing all 17 real (multi-chunk)
            # prompts back-to-back can burst past the API's per-minute
            # token budget faster than a retry-after-429 can recover from
            # -- measured directly while building this script. Spacing
            # requests out keeps the whole run under the ceiling instead
            # of relying on catching the failure after the fact.
            time.sleep(_INTER_QUESTION_DELAY_SECONDS)

        expected = set(entry["expect_article_ids"])
        answer = _ask_with_retry(entry["question"])["answer"]

        if expected:
            retrieval_graded += 1
            top_articles = _top_ranked_articles(entry["question"])
            if top_articles and top_articles[0] in expected:
                top1_hits += 1
            if expected & set(top_articles):
                top5_hits += 1

        if entry["type"] == "unanswerable":
            unanswerable_total += 1
            if _is_refusal(answer):
                refusal_hits += 1

        if _answer_is_correct(entry, answer):
            answer_hits += 1

    total = len(eval_set)

    def _pct(n: int, d: int) -> str:
        return f"({round(100 * n / d)}%)" if d else "(n/a)"

    print(f"Questions:         {total}")
    print(f"Top-1 correct:     {top1_hits}/{retrieval_graded}  {_pct(top1_hits, retrieval_graded)}")
    print(f"Top-5 correct:     {top5_hits}/{retrieval_graded}  {_pct(top5_hits, retrieval_graded)}")
    print(f"Answer correct:    {answer_hits}/{total}  {_pct(answer_hits, total)}")
    print(f"Refused correctly: {refusal_hits}/{unanswerable_total}")


if __name__ == "__main__":
    run_eval()
