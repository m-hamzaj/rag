"""Read-only ChromaDB access against day4-rag's already-running collection.

This module deliberately defines ONLY the read paths this project needs
(search_similar_chunks, get_chunks_by_document, ping) -- unlike
day4-rag/rag/db.py, there is no insert_chunks/clear_chunks/create_schema
here at all, not even as unused functions. That's a structural guarantee,
not a convention someone could violate by adding one call later: this
project has no code path capable of mutating day4-rag's corpus, because
the code to do so was never written.

Same collection name and cosine-space assumption as day4-rag/rag/db.py --
CHROMA_COLLECTION_NAME defaults to "chunks", and this collection was
created there with hnsw:space: cosine (Chroma's own default is squared L2,
which would silently make similarity scores meaningless if this module
guessed wrong about it -- it doesn't need to guess, since it never creates
the collection, only reads whatever config day4-rag already used).
"""

import chromadb

from agents.config import CHROMA_COLLECTION_NAME, CHROMA_HOST, CHROMA_PORT


def _get_client() -> chromadb.ClientAPI:
    return chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)


def _get_collection(client: chromadb.ClientAPI | None = None):
    # embedding_function=None -- same reasoning as day4-rag/rag/db.py:
    # embeddings always come from agents/embed.py, never Chroma's own
    # default embedding function. get_collection (not get_or_create), since
    # this project must never bring the collection into existence itself --
    # if it doesn't exist yet, that means day4-rag hasn't been ingested,
    # which is exactly the condition ping() below is meant to surface
    # clearly rather than paper over.
    client = client or _get_client()
    return client.get_collection(name=CHROMA_COLLECTION_NAME, embedding_function=None)


def ping() -> None:
    """Preflight check, called once at the top of agents/cli.py and
    eval_multiagent.py -- fails fast with a clear, actionable message
    rather than letting an unreachable-Chroma error surface for the first
    time three LLM calls deep inside a researcher tool call, where it would
    look like a corpus problem instead of a "the other project's container
    isn't running" problem.
    """
    try:
        _get_collection().count()
    except Exception as exc:
        raise RuntimeError(
            f"Can't reach day4-rag's ChromaDB at {CHROMA_HOST}:{CHROMA_PORT}. "
            "Is `docker compose up -d db` running in day4-rag, with the corpus "
            f"already ingested? (underlying error: {exc})"
        ) from exc


def search_similar_chunks(query_embedding: list[float], top_k: int) -> list[dict]:
    """Same shape and semantics as day4-rag/rag/db.py's function of the
    same name -- see that module's docstring for the distance-to-similarity
    inversion this relies on.
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


def get_chunks_by_document(document_url: str) -> list[dict]:
    """Same shape as day4-rag/rag/db.py's function of the same name."""
    collection = _get_collection()
    result = collection.get(where={"document_url": document_url}, include=["metadatas", "documents"])
    chunks = [
        {"chunk_index": metadata["chunk_index"], "text": text, "document_title": metadata["document_title"]}
        for metadata, text in zip(result["metadatas"], result["documents"])
    ]
    chunks.sort(key=lambda c: c["chunk_index"])
    return chunks
