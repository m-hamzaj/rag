"""Postgres + pgvector storage for article chunks and their embeddings.

Schema creation deliberately uses a raw psycopg connection, not one with
register_vector() already applied. register_vector() needs the `vector`
extension to already exist in the database, so calling it before `CREATE
EXTENSION vector` has ever run raises "vector type not found" on a brand
new database -- a real chicken-and-egg bug from an earlier build of this
project. The fix: schema creation never registers the vector type at all
(it doesn't need to -- DDL doesn't touch vector values), and only
connections used for actually reading/writing an `embedding` column call
register_vector().
"""

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector

from rag.config import DATABASE_URL, EMBEDDING_DIM


def create_schema() -> None:
    """Creates the vector extension and chunks table if they don't exist
    yet. Safe to call on every startup -- idempotent via IF NOT EXISTS."""
    with psycopg.connect(DATABASE_URL, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id BIGSERIAL PRIMARY KEY,
                document_url TEXT NOT NULL,
                document_title TEXT NOT NULL,
                chunk_index INT NOT NULL,
                text TEXT NOT NULL,
                embedding VECTOR({EMBEDDING_DIM}) NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                UNIQUE (document_url, chunk_index)
            )
            """
        )


def get_connection() -> psycopg.Connection:
    """A connection with the vector type registered, for any query that
    reads or writes an `embedding` column. create_schema() deliberately
    does not use this -- see the module docstring."""
    conn = psycopg.connect(DATABASE_URL, autocommit=True)
    register_vector(conn)
    return conn


def clear_chunks() -> None:
    """Wipes every stored chunk. Called before a fresh ingest run so
    re-ingesting doesn't leave stale chunks behind from articles that
    changed or dropped out of Day 1 since the last run."""
    with psycopg.connect(DATABASE_URL, autocommit=True) as conn:
        conn.execute("TRUNCATE TABLE chunks")


def insert_chunks(records: list[dict]) -> None:
    """records: [{"document_url", "document_title", "chunk_index", "text", "embedding"}, ...].
    Upserts on (document_url, chunk_index), so re-ingesting the same
    article's chunks overwrites the old ones instead of duplicating them.

    Each embedding is wrapped in pgvector.Vector before being sent --
    register_vector() only teaches psycopg to adapt a Vector (or a numpy
    array) into the `vector` type, not a bare Python list. A bare list
    happens to still work here, since the target column's declared type
    gives Postgres a coercion hint an INSERT can use -- but see
    search_similar_chunks() below for where an unwrapped list actually
    breaks, and this wraps things here too rather than relying on that
    asymmetry.
    """
    if not records:
        return
    wrapped = [{**r, "embedding": Vector(r["embedding"])} for r in records]
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO chunks (document_url, document_title, chunk_index, text, embedding)
                VALUES (%(document_url)s, %(document_title)s, %(chunk_index)s, %(text)s, %(embedding)s)
                ON CONFLICT (document_url, chunk_index) DO UPDATE SET
                    text = EXCLUDED.text,
                    embedding = EXCLUDED.embedding,
                    document_title = EXCLUDED.document_title
                """,
                wrapped,
            )


def search_similar_chunks(query_embedding: list[float], top_k: int) -> list[dict]:
    """Returns up to top_k chunks ordered by similarity to query_embedding,
    most similar first. Each result carries a `similarity` field in
    [-1, 1] (1 = identical direction) -- pgvector's `<=>` operator is
    cosine *distance* (1 - similarity), inverted here once rather than
    making every caller remember to do it themselves.

    query_embedding is wrapped in pgvector.Vector before being sent as a
    query parameter. Found by hand, running a real query: a bare Python
    list has no target-column type to hint at, so psycopg dumps it as a
    plain `double precision[]` -- and Postgres has no `<=>` operator
    between `vector` and `double precision[]`, so an unwrapped list fails
    with "operator does not exist" the moment this runs for real (ingest's
    INSERTs succeeded regardless, which is what made this easy to miss).

    Brute-force (no ANN index): the corpus is a few thousand chunks at
    most, well within the range where an exact `ORDER BY ... LIMIT`
    outperforms tuning an ivfflat index's `lists` parameter for a table
    this small.
    """
    vector = Vector(query_embedding)
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT document_url, document_title, chunk_index, text,
                       1 - (embedding <=> %s) AS similarity
                FROM chunks
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (vector, vector, top_k),
            )
            columns = [desc[0] for desc in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]


def count_chunks() -> int:
    with psycopg.connect(DATABASE_URL, autocommit=True) as conn:
        return conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
