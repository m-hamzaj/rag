import httpx
import pytest

from rag import generate as generate_module
from rag.generate import _build_prompt, _build_related_prompt, _cited_indices, generate_answer

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


def test_system_prompt_tells_the_model_excerpts_are_untrusted_not_instructions():
    prompt = generate_module._SYSTEM_PROMPT.lower()
    assert "untrusted" in prompt
    assert "never" in prompt and "instructions" in prompt


def test_system_prompt_names_the_injection_pattern_it_must_refuse():
    # Not just a vague "be careful" -- names the actual attack shape
    # (a scraped article claiming to be a system/developer message) so the
    # model has a concrete pattern to recognize, not just a mood.
    prompt = generate_module._SYSTEM_PROMPT.lower()
    assert "ignore previous instructions" in prompt or "role-play" in prompt


def test_build_prompt_wraps_each_excerpt_in_delimiters():
    prompt = _build_prompt("What?", _CHUNKS)
    # </excerpt> is unambiguous (only the closing tags -- the trailing
    # reminder sentence only mentions the opening form in prose).
    assert prompt.count("</excerpt>") == len(_CHUNKS)
    # The chunk text must be INSIDE the tags, not just present somewhere.
    start = prompt.index("<excerpt>")
    end = prompt.index("</excerpt>")
    assert "chunk one text" in prompt[start:end]


def test_build_prompt_reminds_the_model_excerpts_are_untrusted_right_before_the_question():
    prompt = _build_prompt("What?", _CHUNKS)
    reminder_pos = prompt.lower().index("untrusted reference material")
    question_pos = prompt.index("Question: What?")
    last_excerpt_pos = prompt.rindex("</excerpt>")
    # Sandwiched: after the last excerpt, before the question -- not just
    # present anywhere in the prompt.
    assert last_excerpt_pos < reminder_pos < question_pos


# --- The related-tier prompt: caveated background answer, used when
# retrieval found nothing that clears SIMILARITY_THRESHOLD but something
# clears RELATED_SIMILARITY_THRESHOLD.

def test_related_system_prompt_requires_the_caveat_and_forbids_inventing_specifics():
    prompt = generate_module._RELATED_SYSTEM_PROMPT.lower()
    assert "does not directly answer" in prompt or "doesn't directly" in prompt
    assert "invent" in prompt or "do not invent" in prompt


def test_related_system_prompt_still_carries_the_injection_guard():
    # Hardening against embedded instructions must not be dropped just
    # because this is the "softer" prompt.
    prompt = generate_module._RELATED_SYSTEM_PROMPT.lower()
    assert "untrusted" in prompt
    assert "ignore previous instructions" in prompt or "role-play" in prompt


def test_related_system_prompt_still_allows_a_refusal_if_truly_nothing_useful():
    prompt = generate_module._RELATED_SYSTEM_PROMPT.lower()
    assert "i don't know" in prompt


def test_build_related_prompt_labels_excerpts_as_not_a_direct_match():
    prompt = _build_related_prompt("What?", _CHUNKS)
    assert "not a direct match" in prompt.lower()
    assert "chunk one text" in prompt
    assert "Question: What?" in prompt


def test_generate_answer_uses_the_related_prompt_and_system_message_when_flagged(monkeypatch):
    monkeypatch.setattr(generate_module, "GROQ_API_KEY", "fake-key")
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "not directly, but [1]"}}]})

    _patch_groq_response(monkeypatch, handler)

    generate_answer("q?", _CHUNKS, related=True)

    system_msg = captured["body"]["messages"][0]["content"]
    user_msg = captured["body"]["messages"][1]["content"]
    assert system_msg == generate_module._RELATED_SYSTEM_PROMPT
    assert "not a direct match" in user_msg.lower()


def test_generate_answer_uses_the_direct_prompt_by_default(monkeypatch):
    monkeypatch.setattr(generate_module, "GROQ_API_KEY", "fake-key")
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "X [1]"}}]})

    _patch_groq_response(monkeypatch, handler)

    generate_answer("q?", _CHUNKS)  # related not passed -> defaults False

    assert captured["body"]["messages"][0]["content"] == generate_module._SYSTEM_PROMPT


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
