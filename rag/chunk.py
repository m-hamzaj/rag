"""Splits an article's text into overlapping chunks, packed along paragraph
boundaries rather than a blind word count.

Word count, not characters or tokens, is still the sizing unit -- close
enough to token count for a sentence-transformers/Groq pipeline without
needing a real tokenizer just to decide where to cut, and it keeps
chunk_size directly comparable to the word counts already in Day 3's data
quality report.

Day 6 found a real problem with pure word-count slicing: an article
structured as a short numbered list (see RESULTS.md's Q7 finding, a grizzly
bear article with 4 numbered "myths") got its list items scattered across
different chunks purely because of where the 200th word happened to fall,
not because the items were actually too long to keep together. Only the
list item that happened to land in a chunk that also cleared the top-5
similarity cutoff ever reliably made it into an answer -- a retrieval
recall problem created by chunk boundaries that ignored the article's own
structure. Splitting on blank-line paragraph boundaries fixes this
directly: a short paragraph (a numbered fact, a list item) never gets torn
from its own explanation, and several short paragraphs pack into one
chunk together instead of each fighting the corpus alone for a top-5 slot.
"""

import re

from rag.config import CHUNK_OVERLAP_WORDS, CHUNK_SIZE_WORDS

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n+")
_MARKDOWN_HEADER = re.compile(r"^#{1,6}\s", re.MULTILINE)


def _split_paragraphs(text: str) -> list[str]:
    """The atomic units chunk_text refuses to split (except when one alone
    exceeds chunk_size). For text with markdown headers -- the corpus's
    "#### N. Fact: ..." numbered-list style that motivated this rewrite --
    a unit is a header line plus everything under it up to the next
    header, not just a single blank-line-delimited paragraph: the Q7
    article's each numbered fact is a header *plus* one or two follow-up
    paragraphs, and splitting those from their own header defeats the
    point just as much as splitting between facts did. Text with no
    headers at all (most of this corpus) falls back to plain blank-line
    paragraphs, unchanged from before.
    """
    if not _MARKDOWN_HEADER.search(text):
        return [p.strip() for p in _PARAGRAPH_SPLIT.split(text) if p.strip()]

    lines = text.split("\n")
    sections: list[str] = []
    current: list[str] = []
    for line in lines:
        if _MARKDOWN_HEADER.match(line) and current:
            sections.append("\n".join(current).strip())
            current = []
        current.append(line)
    if current:
        sections.append("\n".join(current).strip())
    return [s for s in sections if s]


def _slide_words(words: list[str], chunk_size: int, overlap: int) -> list[str]:
    """The original pure word-count sliding window -- still used, but now
    only as the fallback for a single paragraph too long to fit in one
    chunk on its own (rare: most paragraphs are well under chunk_size).
    """
    step = chunk_size - overlap
    pieces = []
    start = 0
    while start < len(words):
        pieces.append(" ".join(words[start : start + chunk_size]))
        if start + chunk_size >= len(words):
            break
        start += step
    return pieces


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_WORDS, overlap: int = CHUNK_OVERLAP_WORDS) -> list[str]:
    """Packs paragraphs into chunks of up to chunk_size words each, never
    splitting a single paragraph across two chunks unless that paragraph
    alone exceeds chunk_size (in which case it falls back to a word-count
    slide, same as before, for just that one paragraph). Consecutive
    chunks share `overlap` words of trailing context from the previous
    chunk, the same guarantee the old pure word-count version gave.

    Returns [] for empty/whitespace-only text. A final chunk shorter than
    chunk_size is left short rather than padded -- a short last chunk is
    still valid evidence.
    """
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size})")

    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        return []

    chunks: list[str] = []
    current: list[str] = []

    for paragraph in paragraphs:
        para_words = paragraph.split()
        if not para_words:
            continue

        if current and len(current) + len(para_words) > chunk_size:
            chunks.append(" ".join(current))
            # Seed the next chunk with the last `overlap` words of this one
            # for continuity across the boundary -- but only if that seed
            # still leaves room for the whole next paragraph. Splitting a
            # paragraph to make room for the overlap seed would reintroduce
            # the exact problem this rewrite exists to fix, so when the two
            # don't both fit, the overlap loses: the paragraph boundary
            # itself is already a clean, non-orphaning break, which is most
            # of what overlap was protecting against in the old word-slice
            # version anyway.
            tail = current[-overlap:] if overlap else []
            current = tail if len(tail) + len(para_words) <= chunk_size else []

        if len(para_words) > chunk_size:
            # A single paragraph too long to fit in one chunk on its own --
            # flush whatever's pending, then fall back to word-slicing just
            # this paragraph, same algorithm chunk_text used everywhere
            # before Day 6.
            if current:
                chunks.append(" ".join(current))
                current = []
            chunks.extend(_slide_words(para_words, chunk_size, overlap))
            continue

        current.extend(para_words)

    if current:
        chunks.append(" ".join(current))

    return chunks
