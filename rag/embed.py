"""Turns text into vectors for storage and retrieval.

Uses fastembed -- a small ONNX-based library, not sentence-transformers/
torch directly -- so this stays local and free with no API key, and with
a much smaller Docker image than pulling in torch would need for the same
model weights. The model downloads once (baked into the image) and runs
entirely on CPU.
"""

from fastembed import TextEmbedding

from rag.config import EMBEDDING_MODEL_NAME, FASTEMBED_CACHE_DIR

_model: TextEmbedding | None = None


def _get_model() -> TextEmbedding:
    """Loaded once per process, not once per call -- loading the ONNX
    model has real (if not huge) startup cost, and every call in a given
    process embeds into the same vector space anyway. cache_dir is fixed
    and explicit (see config.py) so the Dockerfile's build-time download
    and the runtime load are guaranteed to agree on where the weights are.
    """
    global _model
    if _model is None:
        _model = TextEmbedding(model_name=EMBEDDING_MODEL_NAME, cache_dir=FASTEMBED_CACHE_DIR)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embeds a batch of texts (e.g. article chunks during ingest) into
    vectors, one per text, in the same order. Returns [] for empty input
    without touching the model at all."""
    if not texts:
        return []
    return [vector.tolist() for vector in _get_model().embed(texts)]


def embed_query(text: str) -> list[float]:
    """Embeds a single piece of text (e.g. a user's question). Goes
    through the exact same model/code path as embed_texts() -- the query
    and the chunks it's compared against must land in the same vector
    space, so there's no separate "query mode" here."""
    return embed_texts([text])[0]
