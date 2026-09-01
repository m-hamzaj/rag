import json

import httpx
import pytest
from groq import APIConnectionError, APIStatusError
from langchain_core.messages import AIMessage

from rag import agent as agent_module
from rag.agent import run_agent

_ARTICLE_A = {
    "document_url": "https://x/a",
    "document_title": "Article A",
    "chunk_index": 0,
    "text": "Article A chunk text",
    "similarity": 0.6,
}
_ARTICLE_B = {
    "document_url": "https://x/b",
    "document_title": "Article B",
    "chunk_index": 0,
    "text": "Article B chunk text",
    "similarity": 0.5,
}


def _tool_call_message(
    name: str, arguments: dict, call_id: str = "call_1", prompt_tokens: int = 100, completion_tokens: int = 20
) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": arguments, "id": call_id, "type": "tool_call"}],
        usage_metadata={
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    )


def _text_message(content: str, prompt_tokens: int = 10, completion_tokens: int = 5) -> AIMessage:
    return AIMessage(
        content=content,
        usage_metadata={
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    )


def _fake_api_error(status_code: int, message: str = "error") -> APIStatusError:
    """A real groq.APIStatusError (the base class RateLimitError,
    BadRequestError, etc. all inherit from, confirmed against the
    installed groq package) -- agent.py's error handling now catches this
    type specifically, not bare Exception, so tests exercise the same
    class, not a stand-in that a narrowed except would no longer catch."""
    fake_response = httpx.Response(status_code, request=httpx.Request("POST", "https://api.groq.com/x"))
    return APIStatusError(message, response=fake_response, body=None)


def _patch_llm_responses(monkeypatch, responses: list):
    """Each call to _call_llm returns the next item in `responses`, in
    order -- an AIMessage to return normally, or an exception instance to
    raise. Mirrors the old httpx-mock sequence pattern, one level up at
    the point agent.py actually calls into LangChain. calls["seen_messages"]
    records the exact message list _call_llm was invoked with each time,
    for tests that need to see what a tool result actually looked like
    once fed back into the conversation."""
    calls = {"count": 0, "seen_messages": []}

    def fake_call_llm(messages):
        calls["seen_messages"].append(list(messages))
        i = calls["count"]
        calls["count"] += 1
        item = responses[i]
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(agent_module, "_call_llm", fake_call_llm)
    return calls


def _patch_corpus(monkeypatch, search_result=None, article_chunks=None):
    monkeypatch.setattr(agent_module.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(agent_module, "embed_query", lambda q: [0.1, 0.2])
    monkeypatch.setattr(
        agent_module, "search_similar_chunks", lambda embedding, top_k: search_result or [_ARTICLE_A, _ARTICLE_B]
    )
    monkeypatch.setattr(agent_module, "get_chunks_by_document", lambda url: article_chunks or [])


def test_normal_flow_search_then_finish(monkeypatch):
    _patch_corpus(monkeypatch)
    _patch_llm_responses(
        monkeypatch,
        [
            _tool_call_message("search_articles", {"query": "grizzly bears"}),
            _tool_call_message("finish", {"answer": "The answer is X."}),
        ],
    )

    result = run_agent("some question?")

    assert result["answer"] == "The answer is X."
    assert result["stopped_reason"] == "finished"
    assert len(result["steps"]) == 2
    assert result["steps"][0]["tool"] == "search_articles"
    assert result["steps"][1]["tool"] == "finish"


def test_citations_include_every_article_touched_by_search(monkeypatch):
    _patch_corpus(monkeypatch)
    _patch_llm_responses(
        monkeypatch,
        [
            _tool_call_message("search_articles", {"query": "q"}),
            _tool_call_message("finish", {"answer": "done"}),
        ],
    )

    result = run_agent("q?")

    urls = {c["url"] for c in result["citations"]}
    assert urls == {"https://x/a", "https://x/b"}


def test_citations_include_article_read_in_full(monkeypatch):
    _patch_corpus(
        monkeypatch,
        search_result=[],
        article_chunks=[{"chunk_index": 0, "text": "full text", "document_title": "Read Article"}],
    )
    _patch_llm_responses(
        monkeypatch,
        [
            _tool_call_message("read_article", {"article_id": "https://x/c"}),
            _tool_call_message("finish", {"answer": "done"}),
        ],
    )

    result = run_agent("q?")

    assert {"title": "Read Article", "url": "https://x/c"} in result["citations"]


def test_read_article_joins_chunks_in_order(monkeypatch):
    _patch_corpus(
        monkeypatch,
        article_chunks=[
            {"chunk_index": 0, "text": "first part", "document_title": "T"},
            {"chunk_index": 1, "text": "second part", "document_title": "T"},
        ],
    )
    calls = _patch_llm_responses(
        monkeypatch,
        [
            _tool_call_message("read_article", {"article_id": "https://x/c"}),
            _tool_call_message("finish", {"answer": "done"}),
        ],
    )

    run_agent("q?")

    # The second _call_llm invocation carries the ToolMessage produced
    # from the first call's read_article result -- this is what the model
    # actually sees, not just what _read_article_tool returns in
    # isolation.
    second_call_messages = calls["seen_messages"][1]
    tool_message = second_call_messages[-1]
    assert tool_message.content == "first part\n\nsecond part"


def test_read_article_truncates_long_articles_to_the_word_cap(monkeypatch):
    # Every tool result stays in the conversation for every later step, so
    # an untruncated article routinely grew a request past Groq's
    # per-request size limit (observed live: 413 "Request too large") --
    # capped specifically to keep that from happening again.
    long_text = " ".join(f"word{i}" for i in range(agent_module._READ_ARTICLE_MAX_WORDS + 500))
    monkeypatch.setattr(
        agent_module,
        "get_chunks_by_document",
        lambda url: [{"chunk_index": 0, "text": long_text, "document_title": "T"}],
    )

    text, title = agent_module._read_article_tool("https://x/a")

    assert len(text.split()) <= agent_module._READ_ARTICLE_MAX_WORDS + 5  # +slack for the truncation marker
    assert "[...article truncated...]" in text
    assert title == "T"


def test_read_article_does_not_truncate_short_articles(monkeypatch):
    short_text = "word " * 100
    monkeypatch.setattr(
        agent_module,
        "get_chunks_by_document",
        lambda url: [{"chunk_index": 0, "text": short_text.strip(), "document_title": "T"}],
    )

    text, title = agent_module._read_article_tool("https://x/a")

    assert "[...article truncated...]" not in text
    assert text == short_text.strip()


def test_max_steps_is_a_hard_limit(monkeypatch):
    _patch_corpus(monkeypatch)
    # Never calls finish -- a different query each time so duplicate-call
    # detection doesn't trip first and mask what this test is checking.
    responses = [_tool_call_message("search_articles", {"query": f"q{i}"}) for i in range(20)]
    _patch_llm_responses(monkeypatch, responses)

    result = run_agent("q?", max_steps=3)

    assert result["stopped_reason"] == "max_steps"
    assert len(result["steps"]) == 3
    assert "3-step limit" in result["answer"]


def test_fallback_answer_includes_full_article_text_not_a_200_char_snippet(monkeypatch):
    # Observed live: a fallback built from a 200-char snippet routinely cut
    # off before the actual fact the question needed, even on runs where
    # the right article had already been read in full -- the hard limit
    # was throwing away evidence the agent genuinely had. The fallback
    # must carry the whole read_article result forward, not a preview.
    long_fact = "x" * 50 + " THE ANSWER IS 42 " + "y" * 500
    _patch_corpus(monkeypatch, article_chunks=[{"chunk_index": 0, "text": long_fact, "document_title": "T"}])
    responses = [_tool_call_message("read_article", {"article_id": "https://x/c"})]
    responses += [_tool_call_message("search_articles", {"query": f"q{i}"}) for i in range(5)]
    _patch_llm_responses(monkeypatch, responses)

    result = run_agent("q?", max_steps=2)

    assert "THE ANSWER IS 42" in result["answer"]


def test_cost_limit_is_a_hard_limit(monkeypatch):
    _patch_corpus(monkeypatch)
    # 1M prompt tokens alone = $0.15 -- two steps of this blows past any
    # small max_cost_usd immediately, real usage numbers, not a stub value.
    expensive = _tool_call_message("search_articles", {"query": "q1"}, prompt_tokens=1_000_000, completion_tokens=0)
    responses = [expensive] + [_tool_call_message("search_articles", {"query": f"q{i}"}) for i in range(2, 10)]
    _patch_llm_responses(monkeypatch, responses)

    result = run_agent("q?", max_steps=8, max_cost_usd=0.01)

    assert result["stopped_reason"] == "cost_limit"
    assert len(result["steps"]) == 1  # stopped after the first, over-budget step
    assert result["cost_usd"] >= 0.01


def test_cost_limit_never_makes_an_extra_paid_call_to_wrap_up(monkeypatch):
    # The fallback answer must be synthesized in plain Python, not via one
    # more LLM call -- otherwise the "hard limit" could itself be the
    # thing that blows the budget.
    _patch_corpus(monkeypatch)
    expensive = _tool_call_message("search_articles", {"query": "q1"}, prompt_tokens=1_000_000, completion_tokens=0)
    calls = _patch_llm_responses(monkeypatch, [expensive])

    run_agent("q?", max_cost_usd=0.01)

    assert calls["count"] == 1


def test_duplicate_tool_call_is_detected_and_stops_the_loop(monkeypatch):
    _patch_corpus(monkeypatch)
    _patch_llm_responses(
        monkeypatch,
        [
            _tool_call_message("search_articles", {"query": "same query"}),
            _tool_call_message("search_articles", {"query": "same query"}),  # identical -- stuck
        ],
    )

    result = run_agent("q?")

    assert result["stopped_reason"] == "duplicate_call"
    assert len(result["steps"]) == 2
    assert "SKIPPED" in result["steps"][1]["result_summary"]


def test_duplicate_detection_is_argument_sensitive_not_just_tool_name(monkeypatch):
    # Calling search_articles twice with DIFFERENT queries is normal,
    # expected behavior for a multi-part question -- must not trip the
    # loop-detector meant for genuinely stuck repetition.
    _patch_corpus(monkeypatch)
    _patch_llm_responses(
        monkeypatch,
        [
            _tool_call_message("search_articles", {"query": "query one"}),
            _tool_call_message("search_articles", {"query": "query two"}),
            _tool_call_message("finish", {"answer": "done"}),
        ],
    )

    result = run_agent("q?")

    assert result["stopped_reason"] == "finished"
    assert len(result["steps"]) == 3


def test_step_log_records_tokens_and_cumulative_cost(monkeypatch):
    _patch_corpus(monkeypatch)
    _patch_llm_responses(
        monkeypatch,
        [
            _tool_call_message("search_articles", {"query": "q"}),
            _tool_call_message("finish", {"answer": "done"}),
        ],
    )

    result = run_agent("q?")

    for step in result["steps"]:
        assert "prompt_tokens" in step
        assert "completion_tokens" in step
        assert "cumulative_cost_usd" in step
    assert result["steps"][1]["cumulative_cost_usd"] >= result["steps"][0]["cumulative_cost_usd"]


def test_usage_and_cost_sum_across_all_steps(monkeypatch):
    _patch_corpus(monkeypatch)
    _patch_llm_responses(
        monkeypatch,
        [
            _tool_call_message("search_articles", {"query": "q"}),  # 100 prompt, 20 completion
            _tool_call_message("finish", {"answer": "done"}),  # another 100 prompt, 20 completion
        ],
    )

    result = run_agent("q?")

    assert result["usage"] == {"prompt_tokens": 200, "completion_tokens": 40}
    assert result["cost_usd"] > 0


def test_llm_is_bound_with_all_three_tools_and_auto_choice(monkeypatch):
    monkeypatch.setattr(agent_module, "GROQ_API_KEY", "fake-key")
    monkeypatch.setattr(agent_module, "_llm", None)
    monkeypatch.setattr(agent_module, "_llm_key_model", None)
    captured = {}

    class _FakeBoundLLM:
        def invoke(self, messages):
            return _tool_call_message("finish", {"answer": "done"})

    class _FakeChatGroq:
        def __init__(self, **kwargs):
            pass

        def bind_tools(self, tools, *, tool_choice=None, **kwargs):
            captured["tool_names"] = {t.name for t in tools}
            captured["tool_choice"] = tool_choice
            captured["kwargs"] = kwargs
            return _FakeBoundLLM()

    monkeypatch.setattr(agent_module, "ChatGroq", _FakeChatGroq)
    _patch_corpus(monkeypatch)

    run_agent("q?")

    assert captured["tool_names"] == {"search_articles", "read_article", "finish"}
    assert captured["tool_choice"] == "auto"
    assert captured["kwargs"].get("parallel_tool_calls") is False


def test_raises_without_api_key(monkeypatch):
    monkeypatch.setattr(agent_module, "GROQ_API_KEY", None)
    with pytest.raises(RuntimeError):
        run_agent("q?")


def test_malformed_tool_call_from_groq_stops_cleanly_instead_of_crashing(monkeypatch):
    # Observed live: Groq can 400 (BadRequestError, code="tool_use_failed")
    # when the model itself emits malformed tool-call JSON -- not a bad
    # request on this end. Must be a clean stop, not an unhandled
    # exception that would kill an entire batch eval run over one bad
    # question.
    _patch_corpus(monkeypatch)
    _patch_llm_responses(monkeypatch, [_fake_api_error(400, "tool_use_failed")])

    result = run_agent("q?")  # must not raise

    assert result["stopped_reason"] == "malformed_tool_call"
    assert "I don't know" in result["answer"] or "Stopped" in result["answer"]


def test_sustained_rate_limit_gets_its_own_distinct_stopped_reason(monkeypatch):
    # A 429 that survives every retry is a completely different situation
    # from a malformed tool call (see agent.py's comment on why these were
    # once conflated as one misleading "model_error" label) -- the two
    # must be distinguishable in the result, not collapsed together.
    monkeypatch.setattr(agent_module, "_RATE_LIMIT_RETRIES", 1)  # keep the test fast
    monkeypatch.setattr(agent_module.time, "sleep", lambda seconds: None)
    _patch_corpus(monkeypatch)

    def always_rate_limited(messages):
        raise _fake_api_error(429, "rate limited")

    monkeypatch.setattr(agent_module, "_get_llm", lambda: type("_L", (), {"invoke": staticmethod(always_rate_limited)})())

    result = run_agent("q?")  # must not raise

    assert result["stopped_reason"] == "rate_limited"


def test_sustained_connection_failure_gets_its_own_distinct_stopped_reason(monkeypatch):
    # Observed live: a raw groq.APIConnectionError (dropped connection, no
    # HTTP response at all -- a genuinely different situation from a 429
    # or a malformed tool call) propagated uncaught and crashed an entire
    # 20-question batch run, losing 18 questions' worth of already-paid-for
    # progress to one transient network hiccup. Must be a clean stop with
    # its own label, not an unhandled exception and not lumped in with
    # "the model generated bad output" (it never got the chance to).
    monkeypatch.setattr(agent_module, "_RATE_LIMIT_RETRIES", 1)  # keep the test fast
    monkeypatch.setattr(agent_module.time, "sleep", lambda seconds: None)
    _patch_corpus(monkeypatch)

    def always_disconnected(messages):
        raise APIConnectionError(request=httpx.Request("POST", "https://api.groq.com/x"))

    monkeypatch.setattr(
        agent_module, "_get_llm", lambda: type("_L", (), {"invoke": staticmethod(always_disconnected)})()
    )

    result = run_agent("q?")  # must not raise

    assert result["stopped_reason"] == "connection_error"


def test_no_tool_call_falls_back_to_plain_text_content(monkeypatch):
    # A real, expected path with tool_choice="auto" (see module docstring
    # for why "required" was tried first and reverted) -- the model
    # concluding with plain text instead of a formal finish() call.
    _patch_corpus(monkeypatch)
    _patch_llm_responses(monkeypatch, [_text_message("just some text")])

    result = run_agent("q?")

    assert result["answer"] == "just some text"
    assert result["stopped_reason"] == "finished"
