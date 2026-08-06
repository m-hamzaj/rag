import httpx
import pytest

from rag import generate as generate_module
from rag.generate import _build_prompt, _cited_indices, generate_answer

_CHUNKS = [
    {"document_url": "https://x/1", "document_title": "Article One", "chunk_index": 0, "text": "chunk one text"},
    {"document_url": "https://x/2", "document_title": "Article Two", "chunk_index": 0, "text": "chunk two text"},
]


def test_build_prompt_numbers_excerpts_from_one():
    prompt = _build_prompt("What?", _CHUNKS)
    assert "[1] (from \"Article One\")" in prompt
    assert "[2] (from \"Article Two\")" in prompt
    assert "chunk one text" in prompt
    assert "Question: What?" in prompt


def test_cited_indices_parses_bracket_markers():
    assert _cited_indices("The answer is X [1] and Y [2].", n_chunks=2) == {1, 2}


def test_cited_indices_ignores_out_of_range_markers():
    # Only 2 chunks provided -- [99] doesn't correspond to a real one.
    assert _cited_indices("See [1] and [99].", n_chunks=2) == {1}


def test_cited_indices_falls_back_to_all_chunks_when_no_markers_found():
    assert _cited_indices("An answer with no citation markers at all.", n_chunks=3) == {1, 2, 3}


def test_cited_indices_deduplicates_repeated_markers():
    assert _cited_indices("[1] again [1] and [2][2]", n_chunks=2) == {1, 2}


def test_generate_answer_raises_without_api_key(monkeypatch):
    monkeypatch.setattr(generate_module, "GROQ_API_KEY", None)
    with pytest.raises(RuntimeError):
        generate_answer("q?", _CHUNKS)


def _patch_groq_response(monkeypatch, handler):
    """Routes generate.py's httpx.post(url, headers=..., json=..., timeout=...)
    through an httpx.MockTransport-backed Client instead of a real network call."""
    transport = httpx.MockTransport(handler)

    def fake_post(url, **kwargs):
        kwargs.pop("timeout", None)
        with httpx.Client(transport=transport) as client:
            return client.post(url, **kwargs)

    monkeypatch.setattr(generate_module.httpx, "post", fake_post)


def test_generate_answer_returns_answer_and_mapped_citations(monkeypatch):
    monkeypatch.setattr(generate_module, "GROQ_API_KEY", "fake-key")
    _patch_groq_response(
        monkeypatch,
        lambda request: httpx.Response(200, json={"choices": [{"message": {"content": "It is X [1]."}}]}),
    )

    result = generate_answer("What is X?", _CHUNKS)

    assert result["answer"] == "It is X [1]."
    assert result["citations"] == [_CHUNKS[0]]


def test_generate_answer_sends_expected_request_body(monkeypatch):
    monkeypatch.setattr(generate_module, "GROQ_API_KEY", "fake-key")
    monkeypatch.setattr(generate_module, "GROQ_MODEL", "some-model")
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        captured["body"] = json.loads(request.content)
        captured["auth"] = request.headers["authorization"]
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok [1]"}}]})

    _patch_groq_response(monkeypatch, handler)

    generate_answer("q?", _CHUNKS)

    assert captured["auth"] == "Bearer fake-key"
    assert captured["body"]["model"] == "some-model"
    assert captured["body"]["messages"][1]["content"].startswith("Excerpts:")
