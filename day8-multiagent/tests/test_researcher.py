from groq import APIConnectionError
import httpx

from agents.researcher import researcher_node
from agents.state import initial_state
from conftest import _fake_api_error, _patch_corpus, _patch_llm_calls, _text_message, _tool_call_message


def test_search_then_plain_text_ends_the_turn(monkeypatch):
    _patch_corpus(monkeypatch)
    _patch_llm_calls(
        monkeypatch,
        [
            _tool_call_message("search_articles", {"query": "grizzly bears"}),
            _text_message("Found two relevant articles about grizzly bears."),
        ],
    )

    result = researcher_node(initial_state("some question?"))

    assert result["stopped_reason"] is None
    assert len(result["research_notes"]) == 2
    assert result["research_notes"][0]["kind"] == "search"
    assert result["research_notes"][1]["query_or_id"] == "(turn summary)"
    assert len(result["step_log"]) == 2
    assert result["total_steps"] == 2


def test_search_touches_every_returned_article_for_citations(monkeypatch):
    _patch_corpus(monkeypatch)
    _patch_llm_calls(
        monkeypatch,
        [_tool_call_message("search_articles", {"query": "q"}), _text_message("done")],
    )

    result = researcher_node(initial_state("q?"))

    assert set(result["articles_touched"]) == {"https://x/a", "https://x/b"}


def test_read_article_appends_full_text_not_a_truncated_preview(monkeypatch):
    long_fact = "x" * 50 + " THE ANSWER IS 42 " + "y" * 500
    _patch_corpus(monkeypatch, article_chunks=[{"chunk_index": 0, "text": long_fact, "document_title": "T"}])
    _patch_llm_calls(
        monkeypatch,
        [_tool_call_message("read_article", {"article_id": "https://x/c"}), _text_message("done")],
    )

    result = researcher_node(initial_state("q?"))

    read_note = next(n for n in result["research_notes"] if n["kind"] == "read")
    assert "THE ANSWER IS 42" in read_note["full_text"]
    assert result["articles_touched"]["https://x/c"] == "T"


def test_read_article_truncates_to_the_word_cap(monkeypatch):
    long_text = " ".join(f"word{i}" for i in range(1700))
    _patch_corpus(monkeypatch, article_chunks=[{"chunk_index": 0, "text": long_text, "document_title": "T"}])

    from agents.researcher import _read_article_tool, _READ_ARTICLE_MAX_WORDS

    text, title = _read_article_tool("https://x/a")

    assert len(text.split()) <= _READ_ARTICLE_MAX_WORDS + 5
    assert "[...article truncated...]" in text


def test_duplicate_call_stops_the_turn(monkeypatch):
    _patch_corpus(monkeypatch)
    _patch_llm_calls(
        monkeypatch,
        [
            _tool_call_message("search_articles", {"query": "same query"}),
            _tool_call_message("search_articles", {"query": "same query"}),
        ],
    )

    result = researcher_node(initial_state("q?"))

    assert result["stopped_reason"] == "duplicate_call"
    assert "SKIPPED" in result["step_log"][-1]["result_summary"]


def test_duplicate_detection_is_argument_sensitive(monkeypatch):
    _patch_corpus(monkeypatch)
    _patch_llm_calls(
        monkeypatch,
        [
            _tool_call_message("search_articles", {"query": "query one"}),
            _tool_call_message("search_articles", {"query": "query two"}),
            _text_message("done"),
        ],
    )

    result = researcher_node(initial_state("q?"))

    assert result["stopped_reason"] is None
    assert result["total_steps"] == 3


def test_local_turn_cap_stops_without_being_a_hard_stop(monkeypatch):
    import agents.researcher as researcher_module

    monkeypatch.setattr(researcher_module, "RESEARCHER_MAX_STEPS_PER_TURN", 2)
    _patch_corpus(monkeypatch)
    _patch_llm_calls(
        monkeypatch,
        [_tool_call_message("search_articles", {"query": f"q{i}"}) for i in range(10)],
    )

    result = researcher_node(initial_state("q?"))

    # Not a hard stop -- the graph still proceeds to the writer with
    # whatever was gathered (see researcher.py's for/else comment).
    assert result["stopped_reason"] is None
    assert result["total_steps"] == 2


def test_global_step_budget_stops_before_spending_another_call(monkeypatch):
    import agents.researcher as researcher_module

    monkeypatch.setattr(researcher_module, "MAX_TOTAL_STEPS", 1)
    _patch_corpus(monkeypatch)
    calls = _patch_llm_calls(
        monkeypatch,
        [_tool_call_message("search_articles", {"query": f"q{i}"}) for i in range(10)],
    )

    result = researcher_node(initial_state("q?"))

    assert result["stopped_reason"] == "max_steps"
    assert calls["count"] == 1


def test_malformed_tool_call_sets_its_own_stopped_reason(monkeypatch):
    _patch_corpus(monkeypatch)
    _patch_llm_calls(monkeypatch, [_fake_api_error(400, "tool_use_failed")])

    result = researcher_node(initial_state("q?"))  # must not raise

    assert result["stopped_reason"] == "malformed_tool_call"


def test_sustained_rate_limit_sets_its_own_stopped_reason(monkeypatch):
    # call_llm itself is stubbed to always raise here -- this checks
    # researcher_node's own handling of that exception, not agents/llm.py's
    # retry loop (see tests/test_llm.py for that).
    import agents.llm as llm_module

    _patch_corpus(monkeypatch)

    def always_rate_limited(llm, messages):
        raise _fake_api_error(429, "rate limited")

    monkeypatch.setattr(llm_module, "call_llm", always_rate_limited)

    result = researcher_node(initial_state("q?"))  # must not raise

    assert result["stopped_reason"] == "rate_limited"


def test_sustained_connection_failure_sets_its_own_stopped_reason(monkeypatch):
    _patch_corpus(monkeypatch)

    def always_disconnected(llm, messages):
        raise APIConnectionError(request=httpx.Request("POST", "https://api.groq.com/x"))

    import agents.llm as llm_module

    monkeypatch.setattr(llm_module, "call_llm", always_disconnected)

    result = researcher_node(initial_state("q?"))  # must not raise

    assert result["stopped_reason"] == "connection_error"
