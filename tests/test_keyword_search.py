from rag import keyword_search as ks_module
from rag.keyword_search import BM25Index, _tokenize, get_index, reset_index_cache


def test_tokenize_lowercases_and_splits_on_non_alphanumeric():
    assert _tokenize("Paddle-Shaped Tail! 2024") == ["paddle", "shaped", "tail", "2024"]


def _chunks(*texts):
    return [
        {"document_url": f"https://x/{i}", "document_title": f"Doc {i}", "chunk_index": 0, "text": text}
        for i, text in enumerate(texts)
    ]


def test_search_ranks_the_document_actually_containing_the_query_term_first():
    chunks = _chunks(
        "tigers are found in Nepal and India",
        "the weather today is sunny and warm",
        "raised bed gardening improves soil drainage",
    )
    index = BM25Index(chunks)

    hits = index.search("tigers Nepal", top_k=3)

    assert hits[0][0] == 0  # the tiger chunk ranks first


def test_search_returns_no_hits_for_terms_absent_from_the_corpus():
    chunks = _chunks("tigers are found in Nepal and India")
    index = BM25Index(chunks)

    hits = index.search("submarine spaceship", top_k=3)

    assert hits == []


def test_search_scores_exact_rare_term_match_higher_than_a_common_term_alone():
    # "the" appears in every doc (low IDF); "kiwa" only in one (high IDF)
    # -- BM25 should reward the rare, specific term match.
    chunks = _chunks(
        "the kiwa crab lives near hydrothermal vents",
        "the weather in the north is cold in the winter",
    )
    index = BM25Index(chunks)

    hits = index.search("kiwa", top_k=2)

    assert len(hits) == 1
    assert hits[0][0] == 0


def test_search_respects_top_k():
    chunks = _chunks(*[f"tiger tiger tiger number {i}" for i in range(5)])
    index = BM25Index(chunks)

    hits = index.search("tiger", top_k=2)

    assert len(hits) == 2


def test_empty_corpus_search_returns_nothing():
    index = BM25Index([])
    assert index.search("anything", top_k=5) == []


def test_get_index_builds_from_db_and_caches(monkeypatch):
    reset_index_cache()
    calls = []

    def fake_get_all_chunks():
        calls.append(1)
        return _chunks("tigers in Nepal")

    monkeypatch.setattr(ks_module, "get_all_chunks", fake_get_all_chunks)

    index1 = get_index()
    index2 = get_index()

    assert index1 is index2  # built once, reused
    assert len(calls) == 1
    reset_index_cache()
