import pytest
from groq import APIConnectionError
import httpx

from agents.state import initial_state
from agents.writer import writer_node
from conftest import _fake_api_error, _patch_llm_calls, _text_message


def _state_with_notes(question="q?", notes=None, critic_feedback=None):
    state = initial_state(question)
    state["research_notes"] = notes or [
        {"kind": "read", "query_or_id": "https://x/a", "summary": "s", "full_text": "THE ANSWER IS 42"}
    ]
    state["critic_feedback"] = critic_feedback
    return state


def test_writer_drafts_from_research_notes(monkeypatch):
    _patch_llm_calls(monkeypatch, [_text_message("The answer is 42.")])

    result = writer_node(_state_with_notes())

    assert result["draft_answer"] == "The answer is 42."
    assert result["stopped_reason"] is None
    assert result["total_steps"] == 1
    assert result["step_log"][0]["node"] == "writer"


def test_writer_prompt_includes_full_research_note_text(monkeypatch):
    calls = _patch_llm_calls(monkeypatch, [_text_message("draft")])

    writer_node(_state_with_notes())

    prompt = calls["seen_messages"][0][-1].content
    assert "THE ANSWER IS 42" in prompt


def test_writer_includes_critic_feedback_on_a_revision_pass(monkeypatch):
    calls = _patch_llm_calls(monkeypatch, [_text_message("revised draft")])

    writer_node(_state_with_notes(critic_feedback="You missed the second article's claim."))

    prompt = calls["seen_messages"][0][-1].content
    assert "You missed the second article's claim." in prompt


def test_writer_no_research_notes_still_produces_a_draft(monkeypatch):
    _patch_llm_calls(monkeypatch, [_text_message("I don't know.")])

    result = writer_node(_state_with_notes(notes=[]))

    assert result["draft_answer"] == "I don't know."


def test_writer_malformed_tool_call_sets_stopped_reason(monkeypatch):
    _patch_llm_calls(monkeypatch, [_fake_api_error(400, "tool_use_failed")])

    result = writer_node(_state_with_notes())  # must not raise

    assert result["stopped_reason"] == "malformed_tool_call"
    assert result.get("draft_answer") is None


def test_writer_sustained_rate_limit_sets_stopped_reason(monkeypatch):
    import agents.llm as llm_module

    def always_rate_limited(llm, messages):
        raise _fake_api_error(429)

    monkeypatch.setattr(llm_module, "call_llm", always_rate_limited)

    result = writer_node(_state_with_notes())  # must not raise

    assert result["stopped_reason"] == "rate_limited"


def test_writer_sustained_connection_failure_sets_stopped_reason(monkeypatch):
    import agents.llm as llm_module

    def always_disconnected(llm, messages):
        raise APIConnectionError(request=httpx.Request("POST", "https://api.groq.com/x"))

    monkeypatch.setattr(llm_module, "call_llm", always_disconnected)

    result = writer_node(_state_with_notes())  # must not raise

    assert result["stopped_reason"] == "connection_error"


def test_writer_raises_without_api_key(monkeypatch):
    import agents.writer as writer_module

    monkeypatch.setattr(writer_module, "GROQ_API_KEY", None)

    with pytest.raises(RuntimeError):
        writer_node(_state_with_notes())
