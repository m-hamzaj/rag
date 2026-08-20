"""Ties retrieval and generation together -- the actual "ask a question,
get an answer with citations, or a refusal" entrypoint the CLI calls.
"""

from rag.generate import generate_answer
from rag.retrieve import retrieve

NO_ANSWER = "I don't know."

# Every ask() return dict carries "usage" now (Day 6 -- eval.py needs real
# token counts for $/run cost tracking). The two fast-refusal paths below
# never call the LLM, so their usage is genuinely zero, not "unknown" --
# this is that zero, not a placeholder.
_ZERO_USAGE = {"prompt_tokens": 0, "completion_tokens": 0}

# Prepended to a related-tier answer so it's unmistakable at a glance --
# not just relying on the model remembering its own instruction to say so
# in prose. Same reasoning as SIMILARITY_THRESHOLD itself: don't leave a
# thing this important to a prompt instruction alone.
_RELATED_PREFIX = "*Related, not a direct answer:* "


def _is_refusal(answer: str) -> bool:
    normalized = answer.strip().rstrip(".").lower()
    return normalized in ("i don't know", "i do not know")


def _dedupe_citations(chunks: list[dict]) -> list[dict]:
    """Multiple cited chunks can come from the same article -- citations
    should list each source article once, not once per chunk, in the
    order they were first cited."""
    seen = set()
    citations = []
    for c in chunks:
        key = c["document_url"]
        if key in seen:
            continue
        seen.add(key)
        citations.append({"title": c["document_title"], "url": c["document_url"]})
    return citations


def ask(question: str) -> dict:
    """Returns {"answer": str, "citations": [{"title", "url"}, ...], "usage": {"prompt_tokens", "completion_tokens"}}.

    Three tiers, checked in order:
      1. ACCEPTED chunks (clear SIMILARITY_THRESHOLD) -- answer directly.
         Still refuses ("I don't know.") if the LLM itself decides these
         plausible-looking chunks don't actually answer the question.
      2. No accepted chunks, but RELATED ones (clear the lower
         RELATED_SIMILARITY_THRESHOLD) -- topically close but not a
         direct match. Answered with a caveated background reply
         (prefixed so the distinction is visible, not just stated in
         prose the model could omit), grounded only in those chunks.
         Still refuses if even the related chunks turn out useless.
      3. Neither -- nothing in the corpus is even topically close.
         Refused before the LLM is ever called at all. This is the only
         remaining case where the refusal is enforced by retrieval
         rather than left to the model's judgment.
    """
    chunks = retrieve(question)

    if chunks["accepted"]:
        result = generate_answer(question, chunks["accepted"])
        if _is_refusal(result["answer"]):
            return {"answer": result["answer"], "citations": [], "usage": result["usage"]}
        return {
            "answer": result["answer"],
            "citations": _dedupe_citations(result["citations"]),
            "usage": result["usage"],
        }

    if chunks["related"]:
        result = generate_answer(question, chunks["related"], related=True)
        if _is_refusal(result["answer"]):
            return {"answer": result["answer"], "citations": [], "usage": result["usage"]}
        return {
            "answer": _RELATED_PREFIX + result["answer"],
            "citations": _dedupe_citations(result["citations"]),
            "usage": result["usage"],
        }

    return {"answer": NO_ANSWER, "citations": [], "usage": dict(_ZERO_USAGE)}
