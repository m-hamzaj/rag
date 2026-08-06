"""Calls Groq to write an answer from retrieved chunks, and parses which
citation markers ([1], [2], ...) the answer actually used -- citations
reflect what the model says it relied on, not just everything retrieval
happened to hand it.
"""

import re

import httpx

from rag.config import GROQ_API_KEY, GROQ_MODEL

_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

_SYSTEM_PROMPT = (
    "You answer questions using ONLY the numbered excerpts provided below. "
    "Cite the excerpt number(s) you used for each claim, like [1] or [2][3]. "
    'If the excerpts don\'t contain enough information to answer the question, '
    'reply with exactly "I don\'t know." and nothing else -- do not guess or '
    "use outside knowledge."
)


def _build_prompt(question: str, chunks: list[dict]) -> str:
    excerpts = "\n\n".join(
        f"[{i + 1}] (from \"{c['document_title']}\")\n{c['text']}" for i, c in enumerate(chunks)
    )
    return f"Excerpts:\n{excerpts}\n\nQuestion: {question}"


def _cited_indices(answer: str, n_chunks: int) -> set[int]:
    """Parses [N] markers from the answer text, keeping only ones that
    correspond to a real, provided chunk (1-indexed). An answer with no
    markers at all falls back to citing every retrieved chunk -- silently
    dropping attribution the reader has no way to recover is worse than
    over-citing.
    """
    found = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    valid = {n for n in found if 1 <= n <= n_chunks}
    if not valid:
        return set(range(1, n_chunks + 1))
    return valid


def generate_answer(question: str, chunks: list[dict]) -> dict:
    """Returns {"answer": str, "citations": [chunk, ...]}. `citations` is
    the subset of `chunks` (full dicts, with document_title/document_url)
    that the answer's [N] markers actually point to.
    """
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set")

    prompt = _build_prompt(question, chunks)
    response = httpx.post(
        _GROQ_URL,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
        },
        timeout=30,
    )
    response.raise_for_status()
    answer = response.json()["choices"][0]["message"]["content"].strip()

    cited = _cited_indices(answer, len(chunks))
    citations = [chunks[i - 1] for i in sorted(cited)]
    return {"answer": answer, "citations": citations}
