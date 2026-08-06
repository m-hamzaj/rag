"""Pulls every article from Day 1's Documents API, chunks it, embeds the
chunks, and stores them in pgvector -- the "build the index" half of RAG.
`rag/retrieve.py` reads what this writes.

Day 1's API paginates at 100 items per request max, so this loops through
offsets rather than assuming a single response has everything.
"""

import httpx

from rag.chunk import chunk_text
from rag.config import DOCUMENTS_API_BASE_URL
from rag.db import clear_chunks, create_schema, insert_chunks, list_documents
from rag.embed import embed_texts

_PAGE_SIZE = 100


def fetch_all_documents() -> list[dict]:
    """Pages through Day 1's /documents endpoint and returns every
    document (title, url, text, source)."""
    documents = []
    offset = 0
    with httpx.Client(timeout=30) as client:
        while True:
            response = client.get(
                f"{DOCUMENTS_API_BASE_URL}/documents",
                params={"limit": _PAGE_SIZE, "offset": offset},
            )
            response.raise_for_status()
            batch = response.json()["items"]
            documents.extend(batch)
            if len(batch) < _PAGE_SIZE:
                break
            offset += _PAGE_SIZE
    return documents


def _chunks_for_document(document: dict) -> list[dict]:
    pieces = chunk_text(document["text"])
    if not pieces:
        return []
    embeddings = embed_texts(pieces)
    return [
        {
            "document_url": document["url"],
            "document_title": document["title"],
            "chunk_index": i,
            "text": piece,
            "embedding": embedding,
        }
        for i, (piece, embedding) in enumerate(zip(pieces, embeddings))
    ]


def ingest(full_rebuild: bool = True, quiet: bool = False) -> dict:
    """Fetches every Day 1 document, chunks and embeds it, and stores the
    chunks. full_rebuild=True (the default) wipes existing chunks first,
    so re-running this after Day 3's corpus changes doesn't leave stale
    chunks behind from articles that no longer exist.
    """
    create_schema()
    if full_rebuild:
        clear_chunks()

    documents = fetch_all_documents()
    if not quiet:
        print(f"Fetched {len(documents)} documents from Day 1.")

    total_chunks = 0
    for document in documents:
        records = _chunks_for_document(document)
        insert_chunks(records)
        total_chunks += len(records)
        if not quiet:
            print(f"  [{len(records)} chunks] {document['title']}")

    if not quiet:
        print(f"\nIngested {len(documents)} documents into {total_chunks} chunks.")

    return {"documents": len(documents), "chunks": total_chunks}


def ingest_new_documents(quiet: bool = False) -> dict:
    """Indexes only the Day 1 documents not already in the vector store --
    the fast path for "I just added one article via Day 3's crawler UI,
    index it" (used by the dashboard's own "Index new articles" button),
    instead of re-chunking and re-embedding the entire corpus every time
    the way a full ingest() does. Never wipes anything.

    "Already indexed" is decided by URL, via db.list_documents() -- an
    article Day 3 later re-scrapes with edited text won't be picked up as
    changed by this path; a full ingest() (which upserts every document
    unconditionally) is what refreshes existing articles' content.
    """
    create_schema()
    already_indexed = {d["url"] for d in list_documents()}

    documents = fetch_all_documents()
    new_documents = [d for d in documents if d["url"] not in already_indexed]
    if not quiet:
        print(f"Fetched {len(documents)} documents from Day 1 -- {len(new_documents)} not yet indexed.")

    total_chunks = 0
    for document in new_documents:
        records = _chunks_for_document(document)
        insert_chunks(records)
        total_chunks += len(records)
        if not quiet:
            print(f"  [{len(records)} chunks] {document['title']}")

    if not quiet:
        print(f"\nIndexed {len(new_documents)} new document(s) into {total_chunks} chunks.")

    return {"documents": len(new_documents), "chunks": total_chunks}


if __name__ == "__main__":
    ingest()
