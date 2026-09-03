import pytest
from groq import APIConnectionError
import httpx

from agents.critic import critic_node
from agents.state import initial_state
from conftest import _fake_api_error, _patch_llm_calls, _text_message, _tool_call_message


def _state_with_draft(draft="The answer is 42.", revision_count=0):
    state = initial_state("q?")
    state["research_notes"] = [{"kind": "read", "query_or_id": "a", "summary": "s", "full_text": "42"}]
    state["draft_answer"] = draft
    state["revision_count"] = revision_count
    return state


def test_critic_approves_a_well_supported_draft(monkeypatch):
    _patch_llm_calls(
        monkeypatch, [_tool_call_message("submit_verdict", {"verdict": "approved", "feedback": "well supported"})]
    )

    result = critic_node(_state_with_draft())

    assert result["critic_verdict"] == "approved"
    assert result["verdict_was_defaulted"] is False
    assert result["revision_count"] == 0  # approval never increments


def test_critic_requests_a_revision_and_increments_revision_count(monkeypatch):
    _patch_llm_calls(
        monkeypatch,
        [_tool_call_message("submit_verdict", {"verdict": "revise_writer", "feedback": "missed a fact"})],
    )

    result = critic_node(_state_with_draft(revision_count=0))

    assert result["critic_verdict"] == "revise_writer"
    assert result["critic_feedback"] == "missed a fact"
    assert result["revision_count"] == 1


def test_critic_revision_count_accumulates_across_cycles(monkeypatch):
    _patch_llm_calls(
        monkeypatch,
        [_tool_call_message("submit_verdict", {"verdict": "revise_writer", "feedback": "still missing something"})],
    )

    result = critic_node(_state_with_draft(revision_count=1))

    assert result["revision_count"] == 2


def test_critic_invalid_verdict_value_fails_open_as_approved(monkeypatch):
    # Deterministic (temperature=0) malformed output isn't worth retrying --
    # fail open rather than risk a parse-retry loop (see module docstring).
    _patch_llm_calls(
        monkeypatch, [_tool_call_message("submit_verdict", {"verdict": "maybe", "feedback": "unclear"})]
    )

    result = critic_node(_state_with_draft())

    assert result["critic_verdict"] == "approved"
    assert result["verdict_was_defaulted"] is True


def test_critic_no_tool_call_fails_open_as_approved(monkeypatch):
    _patch_llm_calls(monkeypatch, [_text_message("looks fine to me")])

    result = critic_node(_state_with_draft())

    assert result["critic_verdict"] == "approved"
    assert result["verdict_was_defaulted"] is True
    assert result["critic_feedback"] == "looks fine to me"


def test_critic_malformed_tool_call_from_groq_sets_stopped_reason(monkeypatch):
    _patch_llm_calls(monkeypatch, [_fake_api_error(400, "tool_use_failed")])

    result = critic_node(_state_with_draft())  # must not raise

    assert result["stopped_reason"] == "malformed_tool_call"


def test_critic_sustained_rate_limit_sets_stopped_reason(monkeypatch):
    import agents.llm as llm_module

    def always_rate_limited(llm, messages):
        raise _fake_api_error(429)

    monkeypatch.setattr(llm_module, "call_llm", always_rate_limited)

    result = critic_node(_state_with_draft())  # must not raise

    assert result["stopped_reason"] == "rate_limited"


def test_critic_sustained_connection_failure_sets_stopped_reason(monkeypatch):
    import agents.llm as llm_module

    def always_disconnected(llm, messages):
        raise APIConnectionError(request=httpx.Request("POST", "https://api.groq.com/x"))

    monkeypatch.setattr(llm_module, "call_llm", always_disconnected)

    result = critic_node(_state_with_draft())  # must not raise

    assert result["stopped_reason"] == "connection_error"


def test_critic_raises_without_api_key(monkeypatch):
    import agents.critic as critic_module

    monkeypatch.setattr(critic_module, "GROQ_API_KEY", None)

    with pytest.raises(RuntimeError):
        critic_node(_state_with_draft())


def test_critic_llm_is_bound_with_submit_verdict_tool_and_auto_choice(monkeypatch):
    import agents.critic as critic_module

    captured = {}

    class _FakeBoundLLM:
        def invoke(self, messages):
            return _tool_call_message("submit_verdict", {"verdict": "approved", "feedback": "ok"})

    class _FakeChatGroq:
        def __init__(self, **kwargs):
            pass

        def bind_tools(self, tools, *, tool_choice=None, **kwargs):
            captured["tool_names"] = {t.name for t in tools}
            captured["tool_choice"] = tool_choice
            captured["kwargs"] = kwargs
            return _FakeBoundLLM()

    monkeypatch.setattr(critic_module, "ChatGroq", _FakeChatGroq)

    critic_node(_state_with_draft())

    assert captured["tool_names"] == {"submit_verdict"}
    assert captured["tool_choice"] == "auto"
    assert captured["kwargs"].get("parallel_tool_calls") is False
