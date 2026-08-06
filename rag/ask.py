"""Ties retrieval and generation together -- the actual "ask a question,
get an answer with citations, or a refusal" entrypoint the CLI calls.
"""

from rag.generate import generate_answer
from rag.retrieve import retrieve

NO_ANSWER = "I don't know."


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
    """Returns {"answer": str, "citations": [{"title", "url"}, ...]}.

    Refuses in two different ways, both landing on the same "I don't
    know." with no citations:
      1. Nothing retrieved clears the similarity threshold -- refused
         before the LLM is ever called. This is the primary mechanism:
         the refusal is enforced by retrieval, not left to the model's
         judgment on a prompt instruction it could ignore.
      2. Retrieval found plausible-looking chunks, but the LLM itself
         decided they don't actually answer the question, and said so.
    """
    chunks = retrieve(question)
    if not chunks:
        return {"answer": NO_ANSWER, "citations": []}

    result = generate_answer(question, chunks)
    if _is_refusal(result["answer"]):
        return {"answer": result["answer"], "citations": []}

    return {"answer": result["answer"], "citations": _dedupe_citations(result["citations"])}
