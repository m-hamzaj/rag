"""Splits an article's text into overlapping, fixed-size word chunks.

Word count, not characters or tokens, is the unit -- close enough to
token count for a sentence-transformers/Groq pipeline without needing a
real tokenizer just to decide where to cut, and it keeps chunk_size directly
comparable to the word counts already in Day 3's data quality report.
"""

from rag.config import CHUNK_OVERLAP_WORDS, CHUNK_SIZE_WORDS


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_WORDS, overlap: int = CHUNK_OVERLAP_WORDS) -> list[str]:
    """Slides a chunk_size-word window over the text, advancing by
    (chunk_size - overlap) words each step, so consecutive chunks share
    `overlap` words -- a sentence describing something at a chunk boundary
    isn't split away from context that would help retrieval find it.

    Returns [] for empty/whitespace-only text. The last chunk is whatever
    words remain, even if shorter than chunk_size -- padding it wouldn't
    add real content, and a short final chunk is still valid evidence.
    """
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size})")

    words = text.split()
    if not words:
        return []

    step = chunk_size - overlap
    chunks = []
    start = 0
    while start < len(words):
        chunks.append(" ".join(words[start : start + chunk_size]))
        if start + chunk_size >= len(words):
            break
        start += step
    return chunks
