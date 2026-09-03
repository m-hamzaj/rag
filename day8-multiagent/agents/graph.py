"""Wires researcher -> writer -> critic -> (writer | finalize) into a
LangGraph StateGraph, with two independent termination layers -- see
"Guaranteed termination" below.

WHY THREE ROLES, NOT ONE LOOP: day4-rag/rag/agent.py's single agent
searches, reads, and answers itself, in one loop. This project splits that
into three specialized calls instead -- a researcher that only gathers
evidence, a writer that only drafts prose from that evidence, and a critic
that checks the draft against the evidence before it goes out. The bet
(unmeasured until eval_multiagent.py runs against day4-rag/data/
agent_eval_set.json's questions) is that separating "did I find the right
facts" from "did I write them up correctly" catches mistakes a single
model juggling both jobs at once would miss -- at the cost of more Groq
calls per question, same kind of trade Day 7 itself made against plain RAG.

WHY THE CRITIC ONLY EVER ROUTES BACK TO THE WRITER (no researcher-recall
path): the user's explicit choice for v1, weighing "truer to three
distinct roles" (the critic could say the EVIDENCE is insufficient, not
just the prose) against "simpler graph, easier to test and reason about."
If a draft is unsupported, that's treated as a writing problem to fix
against the SAME research notes, never a reason to send the graph back to
the researcher. A researcher-recall path is a plausible v2 once this
simpler version has real eval_multiagent.py numbers behind it.

GUARANTEED TERMINATION -- two independent layers, not one:
  1. In-state counters (total_steps, cost_usd, revision_count), checked by
     the conditional-edge functions below -- the primary mechanism, same
     spirit as agent.py's three hard limits, just checked at graph edges
     instead of inside one loop body.
  2. LangGraph's own recursion_limit, passed to invoke() as a generously
     wide backstop (GRAPH_RECURSION_LIMIT=25 against MAX_TOTAL_STEPS=15) --
     this should never actually fire if the counters above are correct, so
     if it ever does, that's treated as its own distinct stopped_reason
     ("graph_recursion_limit"), not silently folded into "max_steps" --
     a sign the counter logic itself has a bug, worth surfacing as such.
"""

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph

from agents.critic import critic_node
from agents.limits import GRAPH_RECURSION_LIMIT, MAX_COST_USD, MAX_REVISION_CYCLES, MAX_TOTAL_STEPS
from agents.researcher import researcher_node
from agents.state import GraphState, initial_state
from agents.writer import writer_node


def _budget_exhausted(state: GraphState) -> bool:
    return state["total_steps"] >= MAX_TOTAL_STEPS or state["cost_usd"] >= MAX_COST_USD


def _route_after_researcher(state: GraphState) -> str:
    # stopped_reason is set by researcher_node itself on an LLM-call
    # failure, a duplicate-call detection, or exhausting the global budget
    # mid-turn -- any of those means "don't spend a writer call."
    if state.get("stopped_reason"):
        return "finalize"
    return "writer"


def _route_after_writer(state: GraphState) -> str:
    if state.get("stopped_reason"):
        return "finalize"
    if _budget_exhausted(state):
        return "finalize"
    if state["revision_count"] >= MAX_REVISION_CYCLES:
        # Cap already reached by a prior critic verdict -- skip critic
        # entirely for this draft, same discipline as agent.py's cost
        # check: never spend a call (the critic's) purely to enforce a
        # limit that's already known to be hit.
        return "finalize"
    return "critic"


def _route_after_critic(state: GraphState) -> str:
    if state.get("stopped_reason"):
        return "finalize"
    if state["critic_verdict"] == "approved":
        return "finalize"
    # verdict == "revise_writer" -- writer's own edge re-checks the budget
    # and the (just-incremented) revision_count before allowing another
    # critic call, so no redundant check needed here.
    return "writer"


def _fallback_answer(reason: str, research_notes: list[dict]) -> str:
    """Same shape as agent.py's _fallback_answer -- used when no draft
    exists yet (a hard stop fired during the researcher's turn)."""
    if not research_notes:
        return f"I don't know. (Stopped: {reason}, before finding any relevant evidence.)"
    evidence = "\n---\n".join(n["full_text"] for n in research_notes[-3:])
    return f"Stopped before finishing ({reason}). Best-effort answer based on what was found:\n\n{evidence}"


def finalize_node(state: GraphState) -> dict:
    """Pure Python, no LLM call -- assembles the final return shape.
    Determines the DEFINITIVE stopped_reason: an error-type reason a node
    already set takes priority; otherwise this is the single place that
    decides "finished" vs. "revision_limit" vs. "max_steps" vs.
    "cost_limit" by reading the state's own counters, since LangGraph's
    conditional-edge functions can only choose a route, not write back to
    state themselves.
    """
    stopped_reason = state.get("stopped_reason")
    draft = state.get("draft_answer")

    if stopped_reason:
        pass  # a node already set an error-type or duplicate_call reason
    elif state.get("critic_verdict") == "approved" and draft is not None:
        stopped_reason = "finished"
    elif state["revision_count"] >= MAX_REVISION_CYCLES and draft is not None:
        stopped_reason = "revision_limit"
    elif state["total_steps"] >= MAX_TOTAL_STEPS:
        stopped_reason = "max_steps"
    elif state["cost_usd"] >= MAX_COST_USD:
        stopped_reason = "cost_limit"
    else:
        # Shouldn't be reachable given the routing above, but mirrors
        # agent.py's own for/else default rather than leaving this unset.
        stopped_reason = "max_steps"

    if stopped_reason == "finished":
        final_answer = draft
    elif draft is not None:
        final_answer = f"Stopped before final approval ({stopped_reason}). Draft answer:\n\n{draft}"
    else:
        final_answer = _fallback_answer(stopped_reason, state["research_notes"])

    citations = [{"title": title, "url": url} for url, title in state["articles_touched"].items()]

    return {"stopped_reason": stopped_reason, "final_answer": final_answer, "citations": citations}


_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        builder = StateGraph(GraphState)
        builder.add_node("researcher", researcher_node)
        builder.add_node("writer", writer_node)
        builder.add_node("critic", critic_node)
        builder.add_node("finalize", finalize_node)

        builder.add_edge(START, "researcher")
        builder.add_conditional_edges("researcher", _route_after_researcher, ["writer", "finalize"])
        builder.add_conditional_edges("writer", _route_after_writer, ["critic", "finalize"])
        builder.add_conditional_edges("critic", _route_after_critic, ["writer", "finalize"])
        builder.add_edge("finalize", END)

        _graph = builder.compile()
    return _graph


def run_multiagent(question: str) -> dict:
    """Returns {"answer", "citations", "steps", "usage", "cost_usd",
    "stopped_reason"} -- same shape as agent.py's run_agent, for a
    like-for-like comparison in eval_multiagent.py.
    """
    graph = _get_graph()
    try:
        final_state = graph.invoke(initial_state(question), config={"recursion_limit": GRAPH_RECURSION_LIMIT})
    except GraphRecursionError:
        # Should never fire if the in-state counters above are correct --
        # see module docstring. Own distinct label, not folded into
        # "max_steps", specifically so it stands out as a counter-logic
        # bug if it ever shows up in a real step_log.
        return {
            "answer": "Stopped: the graph hit LangGraph's own recursion backstop, which should not "
            "happen if the in-state step/cost/revision counters are working correctly -- "
            "this indicates a bug in this project's own limit-enforcement logic, not a normal stop.",
            "citations": [],
            "steps": [],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            "cost_usd": 0.0,
            "stopped_reason": "graph_recursion_limit",
        }

    return {
        "answer": final_state["final_answer"],
        "citations": final_state["citations"],
        "steps": final_state["step_log"],
        "usage": {
            "prompt_tokens": final_state["total_prompt_tokens"],
            "completion_tokens": final_state["total_completion_tokens"],
        },
        "cost_usd": round(final_state["cost_usd"], 6),
        "stopped_reason": final_state["stopped_reason"],
    }
