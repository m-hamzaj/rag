from rag import retrieve as retrieve_module


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
