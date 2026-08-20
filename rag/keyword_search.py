"""Hand-rolled BM25 keyword search over the same chunks vector search uses.

No external BM25 library -- the formula is short enough to hand-roll, and
that matches this project's existing stance on chunking/embedding/retrieval
(see chunk.py, embed.py, retrieve.py): no LangChain/LlamaIndex, understand
the part worth learning rather than importing it.

Okapi BM25 (Robertson & Sparck Jones), k1=1.5 / b=0.75 -- the defaults
almost every real BM25 implementation ships (Lucene/Elasticsearch included).
Day 6's question is vector vs. keyword vs. hybrid, not a BM25
hyperparameter sweep, so those constants aren't tuned further here.

Brute-force scoring (every doc checked against every query term) rather
than a proper inverted index -- at this corpus's size (low thousands of
chunks), that's a few thousand operations per query, milliseconds, and
building a real inverted index would be optimizing a cost that doesn't
exist yet.
"""

import math
import re
from collections import Counter

from rag.db import get_all_chunks

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_K1 = 1.5
_B = 0.75


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """Built once from the full corpus, reused for every query in a run --
    rebuilding it per-query would mean re-tokenizing every chunk on every
    single question, for no benefit since the corpus doesn't change
    mid-run.
    """

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        doc_tokens = [_tokenize(c["text"]) for c in chunks]
        self.term_freqs = [Counter(toks) for toks in doc_tokens]
        self.doc_len = [len(toks) for toks in doc_tokens]
        self.n_docs = len(chunks)
        self.avg_doc_len = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0

        self.doc_freq: Counter = Counter()
        for toks in doc_tokens:
            self.doc_freq.update(set(toks))

    def _idf(self, term: str) -> float:
        # +1 inside the log keeps this non-negative even for a term that
        # appears in every single document (the classic BM25+ fix for
        # negative IDF on very common terms).
        n_t = self.doc_freq.get(term, 0)
        return math.log((self.n_docs - n_t + 0.5) / (n_t + 0.5) + 1)

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """Returns up to top_k (chunk_index_into_self.chunks, bm25_score)
        pairs, best first. Chunks that score 0 (no query term present at
        all) are dropped rather than padding the result with irrelevant
        chunks just to fill top_k.
        """
        query_terms = _tokenize(query)
        scores = [0.0] * self.n_docs
        for term in set(query_terms):
            idf = self._idf(term)
            if idf <= 0:
                continue
            for i in range(self.n_docs):
                f = self.term_freqs[i].get(term, 0)
                if f == 0:
                    continue
                denom = f + _K1 * (1 - _B + _B * self.doc_len[i] / self.avg_doc_len)
                scores[i] += idf * (f * (_K1 + 1)) / denom

        ranked = sorted(range(self.n_docs), key=lambda i: scores[i], reverse=True)
        return [(i, scores[i]) for i in ranked[:top_k] if scores[i] > 0]


_index_cache: BM25Index | None = None


def get_index() -> BM25Index:
    """Module-level cache -- built lazily on first use, then reused for the
    rest of the process. Each `python eval.py` run is a fresh process, so
    this naturally rebuilds against whatever corpus is live (e.g. after a
    chunk-size change and re-ingest) without needing an explicit cache
    invalidation hook.
    """
    global _index_cache
    if _index_cache is None:
        _index_cache = BM25Index(get_all_chunks())
    return _index_cache


def reset_index_cache() -> None:
    """For tests, and for any long-lived process (the UI) that ingests new
    documents after the index was already built."""
    global _index_cache
    _index_cache = None
