from rag import retrieve as retrieve_module

_UNIT_VEC_X = [1.0, 0.0]
_UNIT_VEC_Y = [0.0, 1.0]
_UNIT_VEC_XY = [0.7071067811865476, 0.7071067811865476]  # 45 degrees between X and Y


def test_retrieve_sorts_into_accepted_and_related_tiers(monkeypatch):
    monkeypatch.setattr(retrieve_module, "embed_query", lambda q: [0.1, 0.2])
    candidates = [
        {"document_url": "https://x/1", "similarity": 0.9},   # accepted
        {"document_url": "https://x/2", "similarity": 0.2},   # related
        {"document_url": "https://x/3", "similarity": 0.4},   # accepted
        {"document_url": "https://x/4", "similarity": 0.05},  # neither
    ]
    monkeypatch.setattr(retrieve_module, "search_similar_chunks", lambda emb, k: candidates)

    result = retrieve_module.retrieve(
        "a question", top_k=4, similarity_threshold=0.35, related_threshold=0.15
    )

    assert [r["document_url"] for r in result["accepted"]] == ["https://x/1", "https://x/3"]
    assert [r["document_url"] for r in result["related"]] == ["https://x/2"]


def test_retrieve_returns_empty_tiers_when_nothing_clears_even_the_related_floor(monkeypatch):
    monkeypatch.setattr(retrieve_module, "embed_query", lambda q: [0.1, 0.2])
    candidates = [{"document_url": "https://x/1", "similarity": 0.1}]
    monkeypatch.setattr(retrieve_module, "search_similar_chunks", lambda emb, k: candidates)

    result = retrieve_module.retrieve("a question", similarity_threshold=0.35, related_threshold=0.2)

    assert result == {"accepted": [], "related": []}


def test_retrieve_boundary_value_lands_in_accepted_not_related(monkeypatch):
    # A chunk exactly AT similarity_threshold must land in accepted, not
    # double-counted into related too -- the boundary belongs to one tier.
    monkeypatch.setattr(retrieve_module, "embed_query", lambda q: [0.1])
    candidates = [{"document_url": "https://x/1", "similarity": 0.35}]
    monkeypatch.setattr(retrieve_module, "search_similar_chunks", lambda emb, k: candidates)

    result = retrieve_module.retrieve("q", similarity_threshold=0.35, related_threshold=0.2)

    assert [r["document_url"] for r in result["accepted"]] == ["https://x/1"]
    assert result["related"] == []


def test_retrieve_passes_top_k_through_to_search(monkeypatch):
    monkeypatch.setattr(retrieve_module, "embed_query", lambda q: [0.1])
    captured = {}

    def fake_search(embedding, top_k):
        captured["top_k"] = top_k
        return []

    monkeypatch.setattr(retrieve_module, "search_similar_chunks", fake_search)

    retrieve_module.retrieve("q", top_k=9)

    assert captured["top_k"] == 9


def test_retrieve_embeds_the_question_and_passes_that_vector_to_search(monkeypatch):
    monkeypatch.setattr(retrieve_module, "embed_query", lambda q: [42.0] if q == "the real question" else [0.0])
    captured = {}

    def fake_search(embedding, top_k):
        captured["embedding"] = embedding
        return []

    monkeypatch.setattr(retrieve_module, "search_similar_chunks", fake_search)

    retrieve_module.retrieve("the real question")

    assert captured["embedding"] == [42.0]


def test_retrieve_rejects_an_unknown_mode(monkeypatch):
    monkeypatch.setattr(retrieve_module, "embed_query", lambda q: [0.1])
    try:
        retrieve_module.retrieve("q", mode="bogus")
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "bogus" in str(exc)


# --- Day 6: keyword mode -----------------------------------------------

class _FakeBM25Index:
    def __init__(self, chunks, hits):
        self.chunks = chunks  # full chunk dicts, with "embedding"
        self._hits = hits  # [(chunk_idx, bm25_score), ...] pre-ranked

    def search(self, query, top_k):
        return self._hits[:top_k]


def _chunk(url, embedding, **extra):
    return {"document_url": url, "document_title": url, "chunk_index": 0, "text": "t", "embedding": embedding, **extra}


def test_keyword_mode_orders_by_bm25_rank_not_by_cosine_similarity(monkeypatch):
    # Chunk B has lower cosine similarity to the query than chunk A, but
    # BM25 ranks B first -- keyword mode's ordering must follow BM25, not
    # silently re-sort by the gating similarity.
    chunk_a = _chunk("https://x/a", _UNIT_VEC_XY)  # closer to query direction
    chunk_b = _chunk("https://x/b", _UNIT_VEC_Y)  # further from query direction
    index = _FakeBM25Index(chunks=[chunk_a, chunk_b], hits=[(1, 9.0), (0, 1.0)])  # B ranked first
    monkeypatch.setattr(retrieve_module, "embed_query", lambda q: _UNIT_VEC_X)
    monkeypatch.setattr(retrieve_module, "get_index", lambda: index)

    result = retrieve_module.retrieve("q", top_k=2, similarity_threshold=-1, related_threshold=-1, mode="keyword")

    assert [c["document_url"] for c in result["accepted"]] == ["https://x/b", "https://x/a"]


def test_keyword_mode_gating_similarity_is_real_cosine_not_bm25_score(monkeypatch):
    chunk_a = _chunk("https://x/a", _UNIT_VEC_X)  # identical direction to query -> similarity 1.0
    index = _FakeBM25Index(chunks=[chunk_a], hits=[(0, 123.456)])  # BM25 score is not 0..1
    monkeypatch.setattr(retrieve_module, "embed_query", lambda q: _UNIT_VEC_X)
    monkeypatch.setattr(retrieve_module, "get_index", lambda: index)

    result = retrieve_module.retrieve("q", top_k=1, similarity_threshold=0.9, related_threshold=0.1, mode="keyword")

    assert len(result["accepted"]) == 1
    assert result["accepted"][0]["similarity"] == 1.0


# --- Day 6: hybrid mode --------------------------------------------------

def test_hybrid_mode_ranks_a_chunk_found_by_both_rankers_above_single_ranker_hits(monkeypatch):
    both = _chunk("https://x/both", _UNIT_VEC_X, similarity=0.5)
    vector_only = _chunk("https://x/vec", _UNIT_VEC_X, similarity=0.9)
    keyword_only = _chunk("https://x/kw", _UNIT_VEC_X)

    monkeypatch.setattr(retrieve_module, "embed_query", lambda q: _UNIT_VEC_X)
    monkeypatch.setattr(
        retrieve_module, "search_similar_chunks",
        lambda emb, k: [both, vector_only],  # "both" ranks 1st in vector too
    )
    index = _FakeBM25Index(
        chunks=[both, keyword_only],
        hits=[(0, 5.0), (1, 1.0)],  # "both" ranks 1st in keyword too
    )
    monkeypatch.setattr(retrieve_module, "get_index", lambda: index)

    result = retrieve_module.retrieve("q", top_k=3, similarity_threshold=-1, related_threshold=-1, mode="hybrid")

    urls = [c["document_url"] for c in result["accepted"]]
    assert urls[0] == "https://x/both"  # rank 1 in both lists beats rank 1 in only one
    assert set(urls) == {"https://x/both", "https://x/vec", "https://x/kw"}


def test_hybrid_mode_computes_real_cosine_for_a_keyword_only_hit(monkeypatch):
    # "kw_only" never appears in vector's own candidate list -- its gating
    # similarity must still be real cosine, computed on the fly, not a
    # missing/zero placeholder.
    kw_only = _chunk("https://x/kw", _UNIT_VEC_X)

    monkeypatch.setattr(retrieve_module, "embed_query", lambda q: _UNIT_VEC_X)
    monkeypatch.setattr(retrieve_module, "search_similar_chunks", lambda emb, k: [])
    index = _FakeBM25Index(chunks=[kw_only], hits=[(0, 5.0)])
    monkeypatch.setattr(retrieve_module, "get_index", lambda: index)

    result = retrieve_module.retrieve("q", top_k=1, similarity_threshold=0.9, related_threshold=0.1, mode="hybrid")

    assert len(result["accepted"]) == 1
    assert result["accepted"][0]["similarity"] == 1.0
