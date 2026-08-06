import httpx
import pytest

from rag import ingest as ingest_module


def _mock_transport(pages: dict):
    """pages: {offset: [items...]} -- returns an httpx.MockTransport that
    serves /documents?limit=...&offset=... from this map."""

    def handler(request: httpx.Request) -> httpx.Response:
        offset = int(request.url.params["offset"])
        items = pages.get(offset, [])
        return httpx.Response(200, json={"items": items})

    return httpx.MockTransport(handler)


def test_fetch_all_documents_stops_after_a_short_page(monkeypatch):
    page0 = [{"title": f"doc{i}", "url": f"https://x/{i}", "text": "t"} for i in range(100)]
    page1 = [{"title": "doc100", "url": "https://x/100", "text": "t"}]
    transport = _mock_transport({0: page0, 100: page1})
    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: real_client(transport=transport, **kw))

    documents = ingest_module.fetch_all_documents()

    assert len(documents) == 101
    assert documents[-1]["title"] == "doc100"


def test_fetch_all_documents_stops_immediately_on_a_single_short_page(monkeypatch):
    page0 = [{"title": "only", "url": "https://x/0", "text": "t"}]
    transport = _mock_transport({0: page0})
    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: real_client(transport=transport, **kw))

    documents = ingest_module.fetch_all_documents()

    assert documents == page0


def test_chunks_for_document_zips_chunks_with_embeddings_and_indexes_them(monkeypatch):
    monkeypatch.setattr(ingest_module, "chunk_text", lambda text: ["chunk a", "chunk b"])
    monkeypatch.setattr(ingest_module, "embed_texts", lambda texts: [[1.0], [2.0]])

    doc = {"url": "https://example.com/post", "title": "A Post", "text": "irrelevant"}
    records = ingest_module._chunks_for_document(doc)

    assert records == [
        {"document_url": "https://example.com/post", "document_title": "A Post", "chunk_index": 0, "text": "chunk a", "embedding": [1.0]},
        {"document_url": "https://example.com/post", "document_title": "A Post", "chunk_index": 1, "text": "chunk b", "embedding": [2.0]},
    ]


def test_chunks_for_document_returns_empty_list_for_document_with_no_chunks(monkeypatch):
    monkeypatch.setattr(ingest_module, "chunk_text", lambda text: [])
    calls = []
    monkeypatch.setattr(ingest_module, "embed_texts", lambda texts: calls.append(texts) or [])

    doc = {"url": "https://example.com/post", "title": "A Post", "text": ""}
    records = ingest_module._chunks_for_document(doc)

    assert records == []
    assert calls == []  # embed_texts never called for a document with no chunks


def test_ingest_wipes_existing_chunks_by_default(monkeypatch):
    cleared = []
    monkeypatch.setattr(ingest_module, "create_schema", lambda: None)
    monkeypatch.setattr(ingest_module, "clear_chunks", lambda: cleared.append(True))
    monkeypatch.setattr(ingest_module, "fetch_all_documents", lambda: [])
    monkeypatch.setattr(ingest_module, "insert_chunks", lambda records: None)

    ingest_module.ingest(quiet=True)

    assert cleared == [True]


def test_ingest_skips_wipe_when_full_rebuild_is_false(monkeypatch):
    cleared = []
    monkeypatch.setattr(ingest_module, "create_schema", lambda: None)
    monkeypatch.setattr(ingest_module, "clear_chunks", lambda: cleared.append(True))
    monkeypatch.setattr(ingest_module, "fetch_all_documents", lambda: [])
    monkeypatch.setattr(ingest_module, "insert_chunks", lambda records: None)

    ingest_module.ingest(full_rebuild=False, quiet=True)

    assert cleared == []


def test_ingest_returns_document_and_chunk_counts(monkeypatch):
    docs = [
        {"url": "https://x/1", "title": "One", "text": "t"},
        {"url": "https://x/2", "title": "Two", "text": "t"},
    ]
    monkeypatch.setattr(ingest_module, "create_schema", lambda: None)
    monkeypatch.setattr(ingest_module, "clear_chunks", lambda: None)
    monkeypatch.setattr(ingest_module, "fetch_all_documents", lambda: docs)
    monkeypatch.setattr(ingest_module, "_chunks_for_document", lambda d: [{"fake": "record"}] * 3)
    inserted = []
    monkeypatch.setattr(ingest_module, "insert_chunks", lambda records: inserted.append(records))

    result = ingest_module.ingest(quiet=True)

    assert result == {"documents": 2, "chunks": 6}
    assert len(inserted) == 2


def test_ingest_quiet_mode_prints_nothing(monkeypatch, capsys):
    monkeypatch.setattr(ingest_module, "create_schema", lambda: None)
    monkeypatch.setattr(ingest_module, "clear_chunks", lambda: None)
    monkeypatch.setattr(ingest_module, "fetch_all_documents", lambda: [])
    monkeypatch.setattr(ingest_module, "insert_chunks", lambda records: None)

    ingest_module.ingest(quiet=True)

    assert capsys.readouterr().out == ""


# --- ingest_new_documents: the dashboard's "Index new articles" button --
# only chunks/embeds documents not already indexed, instead of
# reprocessing the whole corpus the way a full ingest() does.

def test_ingest_new_documents_skips_already_indexed_urls(monkeypatch):
    docs = [
        {"url": "https://x/1", "title": "Old", "text": "t"},
        {"url": "https://x/2", "title": "New", "text": "t"},
    ]
    monkeypatch.setattr(ingest_module, "create_schema", lambda: None)
    monkeypatch.setattr(ingest_module, "list_documents", lambda: [{"title": "Old", "url": "https://x/1"}])
    monkeypatch.setattr(ingest_module, "fetch_all_documents", lambda: docs)
    processed = []
    monkeypatch.setattr(ingest_module, "_chunks_for_document", lambda d: processed.append(d) or [{"fake": "r"}])
    monkeypatch.setattr(ingest_module, "insert_chunks", lambda records: None)

    result = ingest_module.ingest_new_documents(quiet=True)

    assert [d["url"] for d in processed] == ["https://x/2"]
    assert result == {"documents": 1, "chunks": 1}


def test_ingest_new_documents_never_wipes_existing_chunks(monkeypatch):
    # No clear_chunks reference at all in this path -- patch it to raise
    # so any accidental call fails the test loudly.
    def boom():
        raise AssertionError("ingest_new_documents must never wipe anything")

    monkeypatch.setattr(ingest_module, "create_schema", lambda: None)
    monkeypatch.setattr(ingest_module, "clear_chunks", boom)
    monkeypatch.setattr(ingest_module, "list_documents", lambda: [])
    monkeypatch.setattr(ingest_module, "fetch_all_documents", lambda: [])
    monkeypatch.setattr(ingest_module, "insert_chunks", lambda records: None)

    ingest_module.ingest_new_documents(quiet=True)  # must not raise


def test_ingest_new_documents_returns_zero_when_nothing_new(monkeypatch):
    docs = [{"url": "https://x/1", "title": "Old", "text": "t"}]
    monkeypatch.setattr(ingest_module, "create_schema", lambda: None)
    monkeypatch.setattr(ingest_module, "list_documents", lambda: [{"title": "Old", "url": "https://x/1"}])
    monkeypatch.setattr(ingest_module, "fetch_all_documents", lambda: docs)
    monkeypatch.setattr(ingest_module, "insert_chunks", lambda records: None)

    result = ingest_module.ingest_new_documents(quiet=True)

    assert result == {"documents": 0, "chunks": 0}
