from rag import retrieve as retrieve_module


def test_retrieve_filters_out_chunks_below_threshold(monkeypatch):
    monkeypatch.setattr(retrieve_module, "embed_query", lambda q: [0.1, 0.2])
    candidates = [
        {"document_url": "https://x/1", "similarity": 0.9},
        {"document_url": "https://x/2", "similarity": 0.2},
        {"document_url": "https://x/3", "similarity": 0.4},
    ]
    monkeypatch.setattr(retrieve_module, "search_similar_chunks", lambda emb, k: candidates)

    result = retrieve_module.retrieve("a question", top_k=3, similarity_threshold=0.35)

    assert [r["document_url"] for r in result] == ["https://x/1", "https://x/3"]


def test_retrieve_returns_empty_list_when_nothing_clears_threshold(monkeypatch):
    monkeypatch.setattr(retrieve_module, "embed_query", lambda q: [0.1, 0.2])
    candidates = [{"document_url": "https://x/1", "similarity": 0.1}]
    monkeypatch.setattr(retrieve_module, "search_similar_chunks", lambda emb, k: candidates)

    result = retrieve_module.retrieve("a question", similarity_threshold=0.35)

    assert result == []


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
