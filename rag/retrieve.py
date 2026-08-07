"""The hand-written retrieval half of RAG: embed a question, pull the
top-k most similar chunks from pgvector, and sort them into three tiers
based on how good the evidence actually is.

No LangChain/LlamaIndex retriever here -- this is the whole loop: embed,
search, sort.
"""

from rag.config import RELATED_SIMILARITY_THRESHOLD, SIMILARITY_THRESHOLD, TOP_K
from rag.db import search_similar_chunks
from rag.embed import embed_query


def retrieve(
    question: str,
    top_k: int = TOP_K,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    related_threshold: float = RELATED_SIMILARITY_THRESHOLD,
) -> dict:
    """Returns {"accepted": [...], "related": [...]}, both lists of chunk
    dicts sorted best-first.

    accepted -- clears similarity_threshold. Strong enough evidence to
        answer the question directly from.
    related -- below similarity_threshold but at or above
        related_threshold. Topically close but not a direct match --
        rag/ask.py uses these for a caveated "here's related background"
        reply instead of an outright refusal, when accepted is empty.

    Both empty means nothing in the corpus is even topically close --
    the only case rag/ask.py treats as grounds to refuse outright,
    before ever calling the LLM.
    """
    query_embedding = embed_query(question)
    candidates = search_similar_chunks(query_embedding, top_k)
    accepted = [c for c in candidates if c["similarity"] >= similarity_threshold]
    related = [c for c in candidates if related_threshold <= c["similarity"] < similarity_threshold]
    return {"accepted": accepted, "related": related}
