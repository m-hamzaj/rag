"""The hand-written retrieval half of RAG: rank candidate chunks (by
vector similarity, keyword match, or both), then sort them into three
tiers based on how good the evidence actually is.

No LangChain/LlamaIndex retriever here -- this is the whole loop: embed
(or tokenize), search, sort.

Day 6 adds RETRIEVAL_MODE (vector | keyword | hybrid). Whichever mode
picks *which* chunks and in *what order*, accepted/related gating always
runs on real cosine similarity -- SIMILARITY_THRESHOLD and
RELATED_SIMILARITY_THRESHOLD were calibrated against cosine similarity
scores (see config.py), and a BM25 score lives on a completely different,
uncalibrated scale. Rather than invent a second threshold system for
keyword scores, every chunk's gating similarity is always its real cosine
similarity to the query -- computed directly (via the corpus's cached
embeddings) for chunks that keyword search finds but vector search's own
top-k didn't. This keeps the existing, measured thresholds meaningful
regardless of which mode is ranking things.
"""

from rag.config import RELATED_SIMILARITY_THRESHOLD, RETRIEVAL_MODE, SIMILARITY_THRESHOLD, TOP_K
from rag.db import search_similar_chunks
from rag.embed import embed_query
from rag.keyword_search import get_index

# How many candidates each ranker pulls before hybrid fuses them. Wider
# than TOP_K on purpose -- RRF needs real rank lists to fuse, not just
# each ranker's own final top_k, or a chunk one ranker ranks 6th (and so
# never returns) can't be rescued by the other ranker ranking it 1st.
_FUSION_POOL = 15

# Constant from the original Reciprocal Rank Fusion paper (Cormack,
# Clarke & Buettcher, 2009). Not tuned further -- RRF's whole appeal here
# is that it needs no score calibration between rankers; picking a
# different k mostly just reweights how much a low rank in one list can
# still matter, which isn't the thing Day 6 is measuring.
_RRF_K = 60


def _cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if not norm_a or not norm_b:
        return 0.0
    return dot / (norm_a * norm_b)


def _chunk_key(chunk: dict) -> tuple:
    return (chunk["document_url"], chunk["chunk_index"])


def _as_result(chunk: dict, similarity: float) -> dict:
    return {
        "document_url": chunk["document_url"],
        "document_title": chunk["document_title"],
        "chunk_index": chunk["chunk_index"],
        "text": chunk["text"],
        "similarity": similarity,
    }


def _vector_ranked(question: str, top_k: int) -> list[dict]:
    query_embedding = embed_query(question)
    return search_similar_chunks(query_embedding, top_k)  # already carries "similarity"


def _keyword_ranked(question: str, top_k: int) -> list[dict]:
    query_embedding = embed_query(question)  # only used for gating similarity, not for ranking
    index = get_index()
    hits = index.search(question, top_k)
    results = []
    for chunk_idx, _bm25_score in hits:
        chunk = index.chunks[chunk_idx]
        results.append(_as_result(chunk, _cosine(query_embedding, chunk["embedding"])))
    return results


def _hybrid_ranked(question: str, top_k: int) -> list[dict]:
    query_embedding = embed_query(question)
    vector_hits = search_similar_chunks(query_embedding, _FUSION_POOL)
    index = get_index()
    keyword_hits = index.search(question, _FUSION_POOL)

    vector_rank = {_chunk_key(c): rank for rank, c in enumerate(vector_hits, start=1)}
    chunk_lookup: dict[tuple, dict] = {_chunk_key(c): c for c in vector_hits}

    keyword_rank = {}
    for rank, (chunk_idx, _score) in enumerate(keyword_hits, start=1):
        chunk = index.chunks[chunk_idx]
        key = _chunk_key(chunk)
        keyword_rank[key] = rank
        chunk_lookup.setdefault(key, chunk)

    def rrf_score(key: tuple) -> float:
        score = 0.0
        if key in vector_rank:
            score += 1 / (_RRF_K + vector_rank[key])
        if key in keyword_rank:
            score += 1 / (_RRF_K + keyword_rank[key])
        return score

    fused_keys = sorted(vector_rank.keys() | keyword_rank.keys(), key=rrf_score, reverse=True)[:top_k]

    results = []
    for key in fused_keys:
        chunk = chunk_lookup[key]
        # Vector hits already carry a real "similarity"; a keyword-only
        # hit (vector's own top-_FUSION_POOL never surfaced it) needs it
        # computed directly -- see module docstring for why this can't
        # just fall back to the BM25 score instead.
        similarity = chunk.get("similarity")
        if similarity is None:
            similarity = _cosine(query_embedding, chunk["embedding"])
        results.append(_as_result(chunk, similarity))
    return results


_RANKERS = {"vector": _vector_ranked, "keyword": _keyword_ranked, "hybrid": _hybrid_ranked}


def retrieve(
    question: str,
    top_k: int = TOP_K,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    related_threshold: float = RELATED_SIMILARITY_THRESHOLD,
    mode: str | None = None,
) -> dict:
    """Returns {"accepted": [...], "related": [...]}, both lists of chunk
    dicts sorted best-first (by each mode's own ranking -- see module
    docstring for what "best" means per mode).

    accepted -- clears similarity_threshold. Strong enough evidence to
        answer the question directly from.
    related -- below similarity_threshold but at or above
        related_threshold. Topically close but not a direct match --
        rag/ask.py uses these for a caveated "here's related background"
        reply instead of an outright refusal, when accepted is empty.

    Both empty means nothing in the corpus is even topically close --
    the only case rag/ask.py treats as grounds to refuse outright,
    before ever calling the LLM.

    mode defaults to config.RETRIEVAL_MODE (env-overridable); pass it
    explicitly to compare modes within one process, e.g. eval.py.
    """
    mode = mode or RETRIEVAL_MODE
    try:
        ranker = _RANKERS[mode]
    except KeyError:
        raise ValueError(f"Unknown RETRIEVAL_MODE {mode!r}, expected one of {sorted(_RANKERS)}") from None

    candidates = ranker(question, top_k)
    accepted = [c for c in candidates if c["similarity"] >= similarity_threshold]
    related = [c for c in candidates if related_threshold <= c["similarity"] < similarity_threshold]
    return {"accepted": accepted, "related": related}
