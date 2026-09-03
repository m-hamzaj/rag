"""Turns text into vectors -- duplicated verbatim in behavior from
day4-rag/rag/embed.py (see agents/config.py's docstring for why this is a
copy, not an import).

The model name/version here MUST stay identical to day4-rag's: the vectors
already stored in its Chroma collection were produced by this exact model,
and a query embedded into a different vector space wouldn't compare
meaningfully against them -- silently wrong results, not an error.
"""

from fastembed import TextEmbedding

from agents.config import EMBEDDING_MODEL_NAME, FASTEMBED_CACHE_DIR

_model: TextEmbedding | None = None


def _get_model() -> TextEmbedding:
    global _model
    if _model is None:
        _model = TextEmbedding(model_name=EMBEDDING_MODEL_NAME, cache_dir=FASTEMBED_CACHE_DIR)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    return [vector.tolist() for vector in _get_model().embed(texts)]


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
