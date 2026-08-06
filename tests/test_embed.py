import numpy as np

from rag import embed as embed_module


class _FakeModel:
    """Stands in for fastembed's TextEmbedding -- real ONNX inference has
    no place in an offline, fast unit test suite. Returns one fixed-length
    fake vector per input text, keyed by input order."""

    def __init__(self):
        self.calls = []

    def embed(self, documents):
        self.calls.append(list(documents))
        return [np.array([float(len(d)), 0.0, 1.0]) for d in documents]


def test_embed_texts_returns_one_vector_per_text_in_order(monkeypatch):
    fake = _FakeModel()
    monkeypatch.setattr(embed_module, "_get_model", lambda: fake)

    vectors = embed_module.embed_texts(["ab", "abcd"])

    assert vectors == [[2.0, 0.0, 1.0], [4.0, 0.0, 1.0]]


def test_embed_texts_returns_plain_lists_not_numpy_arrays(monkeypatch):
    fake = _FakeModel()
    monkeypatch.setattr(embed_module, "_get_model", lambda: fake)

    vectors = embed_module.embed_texts(["ab"])

    assert isinstance(vectors[0], list)
    assert all(isinstance(x, float) for x in vectors[0])


def test_embed_texts_on_empty_list_does_not_touch_the_model(monkeypatch):
    fake = _FakeModel()
    monkeypatch.setattr(embed_module, "_get_model", lambda: fake)

    result = embed_module.embed_texts([])

    assert result == []
    assert fake.calls == []


def test_embed_query_returns_a_single_vector(monkeypatch):
    fake = _FakeModel()
    monkeypatch.setattr(embed_module, "_get_model", lambda: fake)

    vector = embed_module.embed_query("abc")

    assert vector == [3.0, 0.0, 1.0]


def test_model_is_loaded_lazily_and_cached(monkeypatch):
    load_count = {"n": 0}
    fake = _FakeModel()

    def fake_text_embedding(model_name, cache_dir=None):
        load_count["n"] += 1
        return fake

    monkeypatch.setattr(embed_module, "_model", None)
    monkeypatch.setattr(embed_module, "TextEmbedding", fake_text_embedding)

    embed_module.embed_query("one")
    embed_module.embed_query("two")

    assert load_count["n"] == 1  # loaded once, reused on the second call
