from rag import db as db_module


class _FakeCollection:
    def __init__(self, query_result=None, count_value=0, get_result=None):
        self.upsert_calls = []
        self.query_calls = []
        self._query_result = query_result or {"ids": [[]], "metadatas": [[]], "documents": [[]], "distances": [[]]}
        self._count_value = count_value
        self._get_result = get_result or {"metadatas": []}

    def upsert(self, ids, embeddings, documents, metadatas):
        self.upsert_calls.append({"ids": ids, "embeddings": embeddings, "documents": documents, "metadatas": metadatas})

    def query(self, query_embeddings, n_results, include):
        self.query_calls.append({"query_embeddings": query_embeddings, "n_results": n_results, "include": include})
        return self._query_result

    def get(self, include=None):
        return self._get_result

    def count(self):
        return self._count_value


class _FakeClient:
    def __init__(self, collection=None):
        self.deleted = []
        self.get_or_create_calls = []
        self._collection = collection or _FakeCollection()

    def get_or_create_collection(self, name, metadata=None, embedding_function=None):
        self.get_or_create_calls.append({"name": name, "metadata": metadata, "embedding_function": embedding_function})
        return self._collection

    def delete_collection(self, name):
        self.deleted.append(name)


def test_create_schema_gets_or_creates_the_collection_with_cosine_space(monkeypatch):
    client = _FakeClient()
    monkeypatch.setattr(db_module, "_get_client", lambda: client)

    db_module.create_schema()

    assert len(client.get_or_create_calls) == 1
    call = client.get_or_create_calls[0]
    assert call["name"] == db_module.CHROMA_COLLECTION_NAME
    assert call["metadata"] == {"hnsw:space": "cosine"}


def test_create_schema_never_uses_chromas_default_embedding_function(monkeypatch):
    # Embeddings always come from rag/embed.py -- Chroma's own default
    # embedding function must never silently kick in for a call that
    # omits an explicit embedding.
    client = _FakeClient()
    monkeypatch.setattr(db_module, "_get_client", lambda: client)

    db_module.create_schema()

    assert client.get_or_create_calls[0]["embedding_function"] is None


def test_clear_chunks_deletes_and_recreates_the_collection(monkeypatch):
    client = _FakeClient()
    monkeypatch.setattr(db_module, "_get_client", lambda: client)

    db_module.clear_chunks()

    assert client.deleted == [db_module.CHROMA_COLLECTION_NAME]
    assert len(client.get_or_create_calls) == 1  # recreated after deleting


def test_clear_chunks_does_not_raise_when_collection_does_not_exist_yet(monkeypatch):
    client = _FakeClient()

    def boom(name):
        raise Exception("collection does not exist")

    client.delete_collection = boom
    monkeypatch.setattr(db_module, "_get_client", lambda: client)

    db_module.clear_chunks()  # must not raise


def test_insert_chunks_on_empty_list_does_not_touch_the_db(monkeypatch):
    collection = _FakeCollection()
    monkeypatch.setattr(db_module, "_get_collection", lambda client=None: collection)

    db_module.insert_chunks([])

    assert collection.upsert_calls == []


def test_insert_chunks_upserts_with_ids_derived_from_url_and_chunk_index(monkeypatch):
    collection = _FakeCollection()
    monkeypatch.setattr(db_module, "_get_collection", lambda client=None: collection)
    records = [
        {"document_url": "https://x/a", "document_title": "A", "chunk_index": 0, "text": "hello", "embedding": [0.1, 0.2]},
        {"document_url": "https://x/a", "document_title": "A", "chunk_index": 1, "text": "world", "embedding": [0.3, 0.4]},
    ]

    db_module.insert_chunks(records)

    assert len(collection.upsert_calls) == 1
    call = collection.upsert_calls[0]
    assert call["ids"] == ["https://x/a::0", "https://x/a::1"]
    assert call["embeddings"] == [[0.1, 0.2], [0.3, 0.4]]
    assert call["documents"] == ["hello", "world"]
    assert call["metadatas"] == [
        {"document_url": "https://x/a", "document_title": "A", "chunk_index": 0},
        {"document_url": "https://x/a", "document_title": "A", "chunk_index": 1},
    ]


def test_insert_chunks_reingesting_the_same_chunk_upserts_not_duplicates(monkeypatch):
    # Same document_url + chunk_index twice -> same derived id both times,
    # the same "re-ingest overwrites, doesn't duplicate" guarantee the old
    # Postgres UNIQUE constraint gave.
    collection = _FakeCollection()
    monkeypatch.setattr(db_module, "_get_collection", lambda client=None: collection)
    record = {"document_url": "https://x/a", "document_title": "A", "chunk_index": 0, "text": "v1", "embedding": [0.1]}

    db_module.insert_chunks([record])
    db_module.insert_chunks([{**record, "text": "v2"}])

    ids = [call["ids"][0] for call in collection.upsert_calls]
    assert ids[0] == ids[1]


def test_search_similar_chunks_converts_distance_to_similarity(monkeypatch):
    result = {
        "ids": [["https://x/a::0"]],
        "metadatas": [[{"document_url": "https://x/a", "document_title": "A", "chunk_index": 0}]],
        "documents": [["hello world"]],
        "distances": [[0.13]],
    }
    collection = _FakeCollection(query_result=result)
    monkeypatch.setattr(db_module, "_get_collection", lambda client=None: collection)

    results = db_module.search_similar_chunks([0.1, 0.2], top_k=5)

    assert results == [{
        "document_url": "https://x/a",
        "document_title": "A",
        "chunk_index": 0,
        "text": "hello world",
        "similarity": 1 - 0.13,
    }]


def test_search_similar_chunks_passes_top_k_as_n_results(monkeypatch):
    collection = _FakeCollection()
    monkeypatch.setattr(db_module, "_get_collection", lambda client=None: collection)

    db_module.search_similar_chunks([0.1, 0.2], top_k=7)

    assert collection.query_calls[0]["n_results"] == 7
    assert collection.query_calls[0]["query_embeddings"] == [[0.1, 0.2]]


def test_search_similar_chunks_returns_empty_list_on_empty_collection(monkeypatch):
    empty_result = {"ids": [[]], "metadatas": [[]], "documents": [[]], "distances": [[]]}
    collection = _FakeCollection(query_result=empty_result)
    monkeypatch.setattr(db_module, "_get_collection", lambda client=None: collection)

    results = db_module.search_similar_chunks([0.1, 0.2], top_k=5)

    assert results == []


def test_count_chunks_returns_collection_count(monkeypatch):
    collection = _FakeCollection(count_value=1484)
    monkeypatch.setattr(db_module, "_get_collection", lambda client=None: collection)

    assert db_module.count_chunks() == 1484


def test_list_documents_dedupes_multiple_chunks_from_the_same_article(monkeypatch):
    get_result = {
        "metadatas": [
            {"document_url": "https://x/a", "document_title": "A", "chunk_index": 0},
            {"document_url": "https://x/a", "document_title": "A", "chunk_index": 1},
            {"document_url": "https://x/b", "document_title": "B", "chunk_index": 0},
        ]
    }
    collection = _FakeCollection(get_result=get_result)
    monkeypatch.setattr(db_module, "_get_collection", lambda client=None: collection)

    docs = db_module.list_documents()

    assert sorted(docs, key=lambda d: d["url"]) == [
        {"title": "A", "url": "https://x/a"},
        {"title": "B", "url": "https://x/b"},
    ]


def test_list_documents_returns_empty_list_for_empty_collection(monkeypatch):
    collection = _FakeCollection(get_result={"metadatas": []})
    monkeypatch.setattr(db_module, "_get_collection", lambda client=None: collection)

    assert db_module.list_documents() == []
