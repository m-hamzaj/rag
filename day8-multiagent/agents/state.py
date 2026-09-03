"""Shared state threaded through every node in the graph (agents/graph.py).

One flat TypedDict, not a per-node sub-state -- LangGraph merges each
node's returned dict into this same state, and every hard-limit check
(agents/graph.py's conditional edges) needs to see the whole run's
accumulated cost/step count regardless of which node produced it, the same
way Day 7's single loop kept one running total across every tool call
rather than a separate counter per tool.
"""

from typing import Literal, TypedDict


class ResearchNote(TypedDict):
    kind: Literal["search", "read"]
    query_or_id: str
    summary: str  # short, for step_log/writer-prompt readability
    full_text: str  # full result -- writer.py and finalize's fallback both need the real
    # content, not a truncated summary (same reasoning as agent.py's
    # gathered_evidence using full text, not the 200-char result_summary)


class StepLogEntry(TypedDict):
    step: int
    node: str  # "researcher" | "writer" | "critic"
    tool: str | None
    arguments: dict | None
    result_summary: str
    prompt_tokens: int
    completion_tokens: int
    cumulative_cost_usd: float


class GraphState(TypedDict):
    question: str

    research_notes: list[ResearchNote]
    articles_touched: dict[str, str]  # document_url -> title, for citations

    draft_answer: str | None
    critic_feedback: str | None
    critic_verdict: Literal["approved", "revise_writer"] | None
    # True when the critic replied with plain text instead of calling
    # submit_verdict -- treated as an implicit "approved" (fail open, see
    # agents/critic.py), but logged so a step_log reader can tell a real
    # approval from a defaulted one.
    verdict_was_defaulted: bool

    revision_count: int

    total_steps: int  # incremented by every real LLM call, from any node
    total_prompt_tokens: int
    total_completion_tokens: int
    cost_usd: float

    # (tool_name, sorted-json-args) signatures already called by the
    # researcher this run -- same duplicate-call-means-stuck detection as
    # agent.py, persisted across the whole run (the researcher only runs
    # once per graph invocation in this version, so in practice this never
    # spans more than one researcher visit -- see graph.py's docstring for
    # why there's no researcher-recall path yet).
    seen_research_calls: set[tuple[str, str]]

    step_log: list[StepLogEntry]
    stopped_reason: str | None
    final_answer: str | None
    citations: list[dict] | None


def initial_state(question: str) -> GraphState:
    return GraphState(
        question=question,
        research_notes=[],
        articles_touched={},
        draft_answer=None,
        critic_feedback=None,
        critic_verdict=None,
        verdict_was_defaulted=False,
        revision_count=0,
        total_steps=0,
        total_prompt_tokens=0,
        total_completion_tokens=0,
        cost_usd=0.0,
        seen_research_calls=set(),
        step_log=[],
        stopped_reason=None,
        final_answer=None,
        citations=None,
    )
