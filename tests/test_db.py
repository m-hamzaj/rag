from rag import db as db_module


class _FakeCursor:
    def __init__(self, rows=None, description=None):
        self.executed = []
        self.executemany_calls = []
        self._rows = rows or []
        self.description = description or []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        return self

    def executemany(self, sql, records):
        self.executemany_calls.append((sql, records))

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None


class _FakeConnection:
    def __init__(self, cursor=None):
        self.executed = []
        self._cursor = cursor or _FakeCursor()
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True
        return False

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        return self

    def cursor(self):
        return self._cursor

    def fetchone(self):
        return None

    def close(self):
        self.closed = True


def test_create_schema_never_registers_vector_type(monkeypatch):
    # Regression test: register_vector() needs `CREATE EXTENSION vector`
    # to have already run. Calling it during schema creation itself raises
    # "vector type not found" on a brand-new database -- a real bug from
    # an earlier build. create_schema() must not call register_vector at all.
    conn = _FakeConnection()
    monkeypatch.setattr(db_module.psycopg, "connect", lambda *a, **kw: conn)

    def boom(*a, **kw):
        raise AssertionError("register_vector must not be called during schema creation")

    monkeypatch.setattr(db_module, "register_vector", boom)

    db_module.create_schema()

    sqls = [s for s, _ in conn.executed]
    assert any("CREATE EXTENSION IF NOT EXISTS vector" in s for s in sqls)
    assert any("CREATE TABLE IF NOT EXISTS chunks" in s for s in sqls)


def test_get_connection_registers_vector_type(monkeypatch):
    conn = _FakeConnection()
    registered = []
    monkeypatch.setattr(db_module.psycopg, "connect", lambda *a, **kw: conn)
    monkeypatch.setattr(db_module, "register_vector", lambda c: registered.append(c))

    result = db_module.get_connection()

    assert result is conn
    assert registered == [conn]


def test_insert_chunks_on_empty_list_does_not_touch_the_db(monkeypatch):
    conn = _FakeConnection()
    monkeypatch.setattr(db_module, "get_connection", lambda: conn)

    db_module.insert_chunks([])

    assert conn._cursor.executemany_calls == []


def test_insert_chunks_upserts_on_conflict(monkeypatch):
    conn = _FakeConnection()
    monkeypatch.setattr(db_module, "get_connection", lambda: conn)
    records = [{
        "document_url": "https://example.com/a",
        "document_title": "A",
        "chunk_index": 0,
        "text": "hello",
        "embedding": [0.1, 0.2],
    }]

    db_module.insert_chunks(records)

    assert len(conn._cursor.executemany_calls) == 1
    sql, passed_records = conn._cursor.executemany_calls[0]
    assert "ON CONFLICT (document_url, chunk_index)" in sql
    assert "DO UPDATE SET" in sql
    assert passed_records[0]["document_url"] == records[0]["document_url"]
    assert passed_records[0]["text"] == records[0]["text"]


def test_insert_chunks_wraps_embedding_in_pgvector_vector(monkeypatch):
    # Regression test: register_vector() only adapts pgvector.Vector (or a
    # numpy array) into the `vector` type, not a bare Python list. A bare
    # list still happened to work on INSERT (the column's declared type
    # gives Postgres a coercion hint), but wrapping explicitly here removes
    # the reliance on that coercion instead of only fixing it where it
    # visibly broke (search_similar_chunks, see below).
    conn = _FakeConnection()
    monkeypatch.setattr(db_module, "get_connection", lambda: conn)
    records = [{
        "document_url": "https://example.com/a", "document_title": "A",
        "chunk_index": 0, "text": "hello", "embedding": [0.1, 0.2],
    }]

    db_module.insert_chunks(records)

    _, passed_records = conn._cursor.executemany_calls[0]
    assert isinstance(passed_records[0]["embedding"], db_module.Vector)


def test_search_similar_chunks_returns_dicts_with_similarity(monkeypatch):
    description = [("document_url",), ("document_title",), ("chunk_index",), ("text",), ("similarity",)]
    rows = [("https://example.com/a", "A", 0, "hello world", 0.87)]
    cursor = _FakeCursor(rows=rows, description=description)
    conn = _FakeConnection(cursor=cursor)
    monkeypatch.setattr(db_module, "get_connection", lambda: conn)

    results = db_module.search_similar_chunks([0.1, 0.2], top_k=5)

    assert results == [{
        "document_url": "https://example.com/a",
        "document_title": "A",
        "chunk_index": 0,
        "text": "hello world",
        "similarity": 0.87,
    }]


def test_search_similar_chunks_passes_top_k_as_limit(monkeypatch):
    cursor = _FakeCursor(rows=[], description=[])
    conn = _FakeConnection(cursor=cursor)
    monkeypatch.setattr(db_module, "get_connection", lambda: conn)

    db_module.search_similar_chunks([0.1, 0.2], top_k=7)

    sql, params = cursor.executed[0]
    assert params[-1] == 7
    assert "ORDER BY embedding <=>" in sql


def test_search_similar_chunks_wraps_query_embedding_in_pgvector_vector(monkeypatch):
    # Regression test: a bare Python list has no target-column type to
    # hint at, so psycopg dumps it as a plain `double precision[]` --
    # Postgres has no `<=>` operator between `vector` and that type, so an
    # unwrapped list fails with "operator does not exist" the first time
    # this runs against a real database (ingest's INSERTs succeeding
    # regardless is what made this easy to miss until a real query ran).
    cursor = _FakeCursor(rows=[], description=[])
    conn = _FakeConnection(cursor=cursor)
    monkeypatch.setattr(db_module, "get_connection", lambda: conn)

    db_module.search_similar_chunks([0.1, 0.2], top_k=5)

    sql, params = cursor.executed[0]
    assert isinstance(params[0], db_module.Vector)
    assert isinstance(params[1], db_module.Vector)


def test_clear_chunks_truncates(monkeypatch):
    conn = _FakeConnection()
    monkeypatch.setattr(db_module.psycopg, "connect", lambda *a, **kw: conn)

    db_module.clear_chunks()

    sqls = [s for s, _ in conn.executed]
    assert any("TRUNCATE TABLE chunks" in s for s in sqls)


def test_count_chunks_returns_scalar(monkeypatch):
    conn = _FakeConnection()
    conn.execute = lambda sql, params=None: _FakeCursor(rows=[(42,)])
    monkeypatch.setattr(db_module.psycopg, "connect", lambda *a, **kw: conn)

    assert db_module.count_chunks() == 42
