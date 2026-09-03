import agents.critic as critic_module
import agents.graph as graph_module
import agents.researcher as researcher_module
from agents.graph import run_multiagent
from conftest import _patch_corpus, _patch_llm_calls, _text_message, _tool_call_message


def _approve(feedback="well supported"):
    return _tool_call_message("submit_verdict", {"verdict": "approved", "feedback": feedback})


def _revise(feedback="missing a fact"):
    return _tool_call_message("submit_verdict", {"verdict": "revise_writer", "feedback": feedback})


def test_approves_on_the_first_try(monkeypatch):
    _patch_corpus(monkeypatch)
    _patch_llm_calls(
        monkeypatch,
        [
            _tool_call_message("search_articles", {"query": "q"}),  # researcher
            _text_message("found it"),  # researcher ends turn
            _text_message("The answer is 42."),  # writer
            _approve(),  # critic
        ],
    )

    result = run_multiagent("q?")

    assert result["stopped_reason"] == "finished"
    assert result["answer"] == "The answer is 42."
    assert len(result["steps"]) == 4
    assert [s["node"] for s in result["steps"]] == ["researcher", "researcher", "writer", "critic"]


def test_revise_writer_then_approved(monkeypatch):
    _patch_corpus(monkeypatch)
    _patch_llm_calls(
        monkeypatch,
        [
            _text_message("no search needed"),  # researcher ends turn immediately
            _text_message("Draft one."),  # writer, 1st pass
            _revise("missed the second article"),  # critic -> revise
            _text_message("Draft two, addressing feedback."),  # writer, 2nd pass
            _approve(),  # critic -> approved
        ],
    )

    result = run_multiagent("q?")

    assert result["stopped_reason"] == "finished"
    assert result["answer"] == "Draft two, addressing feedback."
    assert [s["node"] for s in result["steps"]] == ["researcher", "writer", "critic", "writer", "critic"]


def test_revision_cap_stops_without_a_final_critic_call(monkeypatch):
    monkeypatch.setattr(graph_module, "MAX_REVISION_CYCLES", 1)
    _patch_corpus(monkeypatch)
    _patch_llm_calls(
        monkeypatch,
        [
            _text_message("no search needed"),  # researcher
            _text_message("Draft one."),  # writer, 1st pass
            _revise("still not right"),  # critic -> revise_writer, revision_count becomes 1
            _text_message("Draft two."),  # writer, 2nd pass -- MAX_REVISION_CYCLES already hit
        ],
    )

    result = run_multiagent("q?")

    assert result["stopped_reason"] == "revision_limit"
    assert "Draft two." in result["answer"]
    assert [s["node"] for s in result["steps"]] == ["researcher", "writer", "critic", "writer"]  # no 2nd critic call


def test_duplicate_call_in_researcher_skips_straight_to_finalize(monkeypatch):
    _patch_corpus(monkeypatch)
    _patch_llm_calls(
        monkeypatch,
        [
            _tool_call_message("search_articles", {"query": "same"}),
            _tool_call_message("search_articles", {"query": "same"}),  # identical -- stuck
        ],
    )

    result = run_multiagent("q?")

    assert result["stopped_reason"] == "duplicate_call"
    assert [s["node"] for s in result["steps"]] == ["researcher", "researcher"]  # writer/critic never ran


def test_global_step_budget_stops_the_whole_run(monkeypatch):
    monkeypatch.setattr(researcher_module, "MAX_TOTAL_STEPS", 1)
    monkeypatch.setattr(graph_module, "MAX_TOTAL_STEPS", 1)
    _patch_corpus(monkeypatch)
    _patch_llm_calls(
        monkeypatch,
        [_tool_call_message("search_articles", {"query": f"q{i}"}) for i in range(10)],
    )

    result = run_multiagent("q?")

    assert result["stopped_reason"] == "max_steps"
    assert len(result["steps"]) == 1  # writer/critic never got a chance to spend a call


def test_finalize_synthesizes_a_fallback_when_no_draft_exists(monkeypatch):
    # A hard stop during the researcher's own turn (before the writer ever
    # runs) must still produce a real, evidence-based answer -- same
    # "partial beats nothing" philosophy as agent.py's fallback answers.
    monkeypatch.setattr(researcher_module, "MAX_TOTAL_STEPS", 1)
    monkeypatch.setattr(graph_module, "MAX_TOTAL_STEPS", 1)
    _patch_corpus(
        monkeypatch, article_chunks=[{"chunk_index": 0, "text": "THE ANSWER IS 42", "document_title": "T"}]
    )
    _patch_llm_calls(monkeypatch, [_tool_call_message("read_article", {"article_id": "https://x/c"})])

    result = run_multiagent("q?")

    assert result["stopped_reason"] == "max_steps"
    assert "THE ANSWER IS 42" in result["answer"]


def test_critic_defaulted_approval_still_finishes_the_run(monkeypatch):
    # An unparseable critic reply fails open as "approved" (see
    # agents/critic.py) -- the graph should finish cleanly, not stall.
    _patch_corpus(monkeypatch)
    _patch_llm_calls(
        monkeypatch,
        [
            _text_message("no search needed"),
            _text_message("Draft one."),
            _text_message("looks fine, no tool call"),  # critic replies with plain text
        ],
    )

    result = run_multiagent("q?")

    assert result["stopped_reason"] == "finished"
    assert result["answer"] == "Draft one."


def test_recursion_limit_backstop_fires_when_counters_are_broken(monkeypatch):
    # Deliberately break the counter-based stop (a huge MAX_REVISION_CYCLES,
    # so route_after_writer never routes to finalize on its own) to prove
    # the independent recursion_limit backstop still catches a runaway
    # graph -- see agents/graph.py's module docstring on why this is a
    # second, independent termination layer, not a redundant one.
    monkeypatch.setattr(graph_module, "MAX_REVISION_CYCLES", 100_000)
    monkeypatch.setattr(graph_module, "GRAPH_RECURSION_LIMIT", 6)
    _patch_corpus(monkeypatch)

    pattern = [_text_message("no search needed")]  # researcher, once
    # writer draft / critic revise_writer, repeating forever
    responses_cycle = [_text_message("draft"), _revise("still not right")]

    class _InfiniteResponses:
        def __getitem__(self, i):
            if i == 0:
                return pattern[0]
            return responses_cycle[(i - 1) % len(responses_cycle)]

    _patch_llm_calls(monkeypatch, _InfiniteResponses())

    result = run_multiagent("q?")

    assert result["stopped_reason"] == "graph_recursion_limit"
