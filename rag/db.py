"""ChromaDB storage for article chunks and their embeddings.

Switched from Postgres/pgvector to Chroma (team decision after the brief
had already specified pgvector -- noted in the README, not hidden here).
Every function below keeps the exact same name and signature it had under
pgvector (`create_schema`, `clear_chunks`, `insert_chunks`,
`search_similar_chunks`, `count_chunks`), so `rag/ingest.py` and
`rag/retrieve.py` -- which only import these names -- needed zero changes
for the swap. The storage backend is an implementation detail behind this
module; nothing above it should need to know which one is in use.

A single collection ("chunks") explicitly configured for cosine distance
(`hnsw:space: cosine`) -- Chroma's default is squared L2, which would
silently make SIMILARITY_THRESHOLD meaningless (it was tuned against real
cosine similarity scores; see README).
"""

import chromadb

from rag.config import CHROMA_COLLECTION_NAME, CHROMA_HOST, CHROMA_PORT

_COLLECTION_METADATA = {"hnsw:space": "cosine"}


def _get_client() -> chromadb.ClientAPI:
    return chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)


def _get_collection(client: chromadb.ClientAPI | None = None):
    """embedding_function=None throughout this module -- embeddings always
    come from rag/embed.py (the same model used for both chunks and
    queries), never from Chroma's own default embedding function, which
    would otherwise silently kick in for any call that omits an explicit
    embedding.
    """
    client = client or _get_client()
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME, metadata=_COLLECTION_METADATA, embedding_function=None
    )


def create_schema() -> None:
    """Creates the collection if it doesn't exist yet. Safe to call on
    every startup -- get_or_create is idempotent."""
    _get_collection()


def clear_chunks() -> None:
    """Wipes every stored chunk by dropping and recreating the collection.
    Called before a fresh ingest run so re-ingesting doesn't leave stale
    chunks behind from articles that changed or dropped out of Day 1 since
    the last run."""
    client = _get_client()
    try:
        client.delete_collection(CHROMA_COLLECTION_NAME)
    except Exception:
        pass  # nothing to delete yet -- fine on a brand-new database
    _get_collection(client)


def _chunk_id(document_url: str, chunk_index: int) -> str:
    """Deterministic ID from (document_url, chunk_index) -- re-ingesting
    the same article's chunks upserts in place instead of duplicating,
    the same guarantee the old Postgres UNIQUE constraint gave."""
    return f"{document_url}::{chunk_index}"


def insert_chunks(records: list[dict]) -> None:
    """records: [{"document_url", "document_title", "chunk_index", "text", "embedding"}, ...].
    Upserts by (document_url, chunk_index)-derived id -- see _chunk_id.
    """
    if not records:
        return
    collection = _get_collection()
    collection.upsert(
        ids=[_chunk_id(r["document_url"], r["chunk_index"]) for r in records],
        embeddings=[r["embedding"] for r in records],
        documents=[r["text"] for r in records],
        metadatas=[
            {
                "document_url": r["document_url"],
                "document_title": r["document_title"],
                "chunk_index": r["chunk_index"],
            }
            for r in records
        ],
    )


def search_similar_chunks(query_embedding: list[float], top_k: int) -> list[dict]:
    """Returns up to top_k chunks ordered by similarity to query_embedding,
    most similar first. Each result carries a `similarity` field in
    [-1, 1] (1 = identical direction) -- Chroma's cosine-space distance is
    cosine *distance* (1 - similarity), inverted here once rather than
    making every caller remember to do it themselves. Same convention the
    pgvector version used, so retrieve.py's threshold comparison didn't
    need to change.
    """
    collection = _get_collection()
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["metadatas", "documents", "distances"],
    )
    if not result["ids"] or not result["ids"][0]:
        return []

    chunks = []
    for metadata, text, distance in zip(result["metadatas"][0], result["documents"][0], result["distances"][0]):
        chunks.append(
            {
                "document_url": metadata["document_url"],
                "document_title": metadata["document_title"],
                "chunk_index": metadata["chunk_index"],
                "text": text,
                "similarity": 1 - distance,
            }
        )
    return chunks


def count_chunks() -> int:
    return _get_collection().count()


def get_all_chunks() -> list[dict]:
    """Returns every stored chunk, embedding included:
    [{"document_url", "document_title", "chunk_index", "text", "embedding"}, ...].

    Day 6 -- keyword search (rag/keyword_search.py) needs the full corpus's
    text in memory once to build a BM25 index, and hybrid retrieval needs
    real embeddings on hand to compute exact cosine similarity for a
    keyword-only hit that vector search's own top-k never surfaced (see
    rag/retrieve.py's threshold gating). One full-corpus fetch per process,
    not a per-query round trip either way.
    """
    collection = _get_collection()
    result = collection.get(include=["metadatas", "documents", "embeddings"])
    chunks = []
    for metadata, text, embedding in zip(result["metadatas"], result["documents"], result["embeddings"]):
        chunks.append(
            {
                "document_url": metadata["document_url"],
                "document_title": metadata["document_title"],
                "chunk_index": metadata["chunk_index"],
                "text": text,
                "embedding": embedding,
            }
        )
    return chunks


def list_documents() -> list[dict]:
    """Returns one entry per distinct source article currently indexed:
    {"title": ..., "url": ...}. Deduped from chunk metadata rather than
    re-fetching Day 1 -- the corpus this answers questions from is
    whatever's actually indexed, which is what a user deciding what to
    ask needs to see, not a live Day 1 document count that could drift
    from it between ingest runs.
    """
    collection = _get_collection()
    result = collection.get(include=["metadatas"])
    seen = {}
    for metadata in result["metadatas"]:
        seen[metadata["document_url"]] = metadata["document_title"]
    return [{"title": title, "url": url} for url, title in seen.items()]
