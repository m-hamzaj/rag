import pytest

from agents import db as db_module


class _FakeCollection:
    def __init__(self, query_result=None, get_result=None, count_value=0, raise_on_count=False):
        self.query_calls = []
        self.get_calls = []
        self._query_result = query_result or {"ids": [[]], "metadatas": [[]], "documents": [[]], "distances": [[]]}
        self._get_result = get_result or {"metadatas": [], "documents": []}
        self._count_value = count_value
        self._raise_on_count = raise_on_count

    def query(self, query_embeddings, n_results, include):
        self.query_calls.append({"query_embeddings": query_embeddings, "n_results": n_results, "include": include})
        return self._query_result

    def get(self, include=None, where=None):
        self.get_calls.append({"include": include, "where": where})
        return self._get_result

    def count(self):
        if self._raise_on_count:
            raise Exception("connection refused")
        return self._count_value


class _FakeClient:
    def __init__(self, collection=None):
        self.get_calls = []
        self._collection = collection or _FakeCollection()

    def get_collection(self, name, embedding_function=None):
        self.get_calls.append({"name": name, "embedding_function": embedding_function})
        return self._collection


def test_get_collection_never_creates_never_uses_default_embedding_function(monkeypatch):
    # get_collection, not get_or_create_collection -- this project must
    # never bring the collection into existence itself (see module
    # docstring: that would defeat the "structurally can't mutate day4-rag's
    # corpus" guarantee). embedding_function=None, same reasoning as
    # day4-rag/rag/db.py: embeddings always come from agents/embed.py.
    client = _FakeClient()
    monkeypatch.setattr(db_module, "_get_client", lambda: client)

    db_module._get_collection()

    assert client.get_calls[0]["name"] == db_module.CHROMA_COLLECTION_NAME
    assert client.get_calls[0]["embedding_function"] is None


def test_ping_succeeds_silently_when_collection_reachable(monkeypatch):
    collection = _FakeCollection(count_value=125)
    monkeypatch.setattr(db_module, "_get_collection", lambda: collection)

    db_module.ping()  # must not raise


def test_ping_raises_a_clear_actionable_error_when_unreachable(monkeypatch):
    collection = _FakeCollection(raise_on_count=True)
    monkeypatch.setattr(db_module, "_get_collection", lambda: collection)

    with pytest.raises(RuntimeError, match="docker compose up -d db"):
        db_module.ping()


def test_search_similar_chunks_converts_distance_to_similarity(monkeypatch):
    result = {
        "ids": [["https://x/a::0"]],
        "metadatas": [[{"document_url": "https://x/a", "document_title": "A", "chunk_index": 0}]],
        "documents": [["hello world"]],
        "distances": [[0.13]],
    }
    collection = _FakeCollection(query_result=result)
    monkeypatch.setattr(db_module, "_get_collection", lambda: collection)

    results = db_module.search_similar_chunks([0.1, 0.2], top_k=5)

    assert results == [
        {
            "document_url": "https://x/a",
            "document_title": "A",
            "chunk_index": 0,
            "text": "hello world",
            "similarity": 1 - 0.13,
        }
    ]


def test_search_similar_chunks_returns_empty_list_on_empty_collection(monkeypatch):
    collection = _FakeCollection()
    monkeypatch.setattr(db_module, "_get_collection", lambda: collection)

    assert db_module.search_similar_chunks([0.1, 0.2], top_k=5) == []


def test_get_chunks_by_document_filters_by_document_url(monkeypatch):
    collection = _FakeCollection()
    monkeypatch.setattr(db_module, "_get_collection", lambda: collection)

    db_module.get_chunks_by_document("https://x/a")

    assert collection.get_calls[0]["where"] == {"document_url": "https://x/a"}


def test_get_chunks_by_document_sorts_by_chunk_index(monkeypatch):
    get_result = {
        "metadatas": [
            {"document_url": "https://x/a", "document_title": "A", "chunk_index": 2},
            {"document_url": "https://x/a", "document_title": "A", "chunk_index": 0},
            {"document_url": "https://x/a", "document_title": "A", "chunk_index": 1},
        ],
        "documents": ["third", "first", "second"],
    }
    collection = _FakeCollection(get_result=get_result)
    monkeypatch.setattr(db_module, "_get_collection", lambda: collection)

    chunks = db_module.get_chunks_by_document("https://x/a")

    assert [c["text"] for c in chunks] == ["first", "second", "third"]


def test_get_chunks_by_document_returns_empty_list_when_url_not_found(monkeypatch):
    collection = _FakeCollection(get_result={"metadatas": [], "documents": []})
    monkeypatch.setattr(db_module, "_get_collection", lambda: collection)

    assert db_module.get_chunks_by_document("https://x/missing") == []


def test_db_module_has_no_write_paths():
    # Structural guarantee, not a convention -- this project must never be
    # able to mutate day4-rag's corpus, so the write functions simply don't
    # exist here at all (see module docstring).
    for name in ("insert_chunks", "clear_chunks", "create_schema"):
        assert not hasattr(db_module, name)
