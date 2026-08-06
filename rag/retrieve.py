"""The hand-written retrieval half of RAG: embed a question, pull the
top-k most similar chunks from pgvector, and decide whether any of them
are actually good enough evidence to answer from at all.

No LangChain/LlamaIndex retriever here -- this is the whole loop: embed,
search, filter.
"""

from rag.config import SIMILARITY_THRESHOLD, TOP_K
from rag.db import search_similar_chunks
from rag.embed import embed_query


def retrieve(question: str, top_k: int = TOP_K, similarity_threshold: float = SIMILARITY_THRESHOLD) -> list[dict]:
    """Returns the chunks worth answering from: the top_k most similar to
    the question, filtered down to only the ones clearing
    similarity_threshold. An empty return means "nothing in the corpus
    was good enough" -- rag/ask.py treats that as grounds to refuse
    outright, before ever calling the LLM.
    """
    query_embedding = embed_query(question)
    candidates = search_similar_chunks(query_embedding, top_k)
    return [c for c in candidates if c["similarity"] >= similarity_threshold]
