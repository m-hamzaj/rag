"""The critic node: one LLM call, bound to a single tool (submit_verdict),
checking the writer's draft against the researcher's actual notes -- not
against outside knowledge, same "don't trust your own priors over the
retrieved evidence" framing agent.py and generate.py both use for scraped
content.

v1 only ever routes back to the writer (never the researcher) -- the
user's explicit choice for a simpler, easier-to-test graph; a
researcher-recall path is a possible later extension once this is
measured (see agents/graph.py's docstring).

tool_choice="auto", NOT "required" -- even though the critic should always
produce structured output, Day 7 already demonstrated live that Groq's
"required" mode can 400 the whole request rather than degrade gracefully
(agent.py's own docstring). There's no reason to assume the critic model is
exempt from that failure mode just because we'd prefer it always calls the
tool -- so an unparseable reply is handled explicitly below (fail open),
not assumed away.
"""

from groq import APIConnectionError, APIStatusError
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from agents import llm as llm_module
from agents.config import GROQ_API_KEY, GROQ_MODEL
from agents.state import GraphState

# See agents/researcher.py's import comment for why llm_module.call_llm,
# not a direct `from agents.llm import call_llm`.

_SYSTEM_PROMPT = (
    "You are the CRITIC stage of a multi-agent system answering questions about a "
    "corpus of nature/wildlife/gardening articles. You will be given a question, a "
    "researcher's notes, and a writer's draft answer. Check the draft ONLY against "
    "the research notes -- not your own outside knowledge. Call submit_verdict with "
    'verdict="approved" if every claim in the draft is actually supported by the '
    'notes and it directly answers the question, or verdict="revise_writer" if the '
    "draft makes an unsupported claim, misses part of the question, or is unclear -- "
    "in that case, feedback must say specifically what to fix. Always call "
    "submit_verdict; never reply with plain text."
)


class _SubmitVerdictInput(BaseModel):
    verdict: str = Field(description='Either "approved" or "revise_writer".')
    feedback: str = Field(description="Specific, actionable feedback -- required even when approving (briefly why).")


def _submit_verdict_fn(verdict: str, feedback: str) -> str:
    # Never actually invoked -- the node intercepts the tool call before
    # dispatching, same pattern as agent.py's _finish_fn. Exists so the
    # tool is a real, describable part of the bound toolset.
    return f"{verdict}: {feedback}"


_SUBMIT_VERDICT_TOOL = StructuredTool.from_function(
    func=_submit_verdict_fn,
    name="submit_verdict",
    description="Submit your review verdict and feedback on the writer's draft.",
    args_schema=_SubmitVerdictInput,
)

_llm = None
_llm_key_model: tuple[str, str] | None = None


def _get_llm():
    global _llm, _llm_key_model
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set")
    if _llm is None or _llm_key_model != (GROQ_API_KEY, GROQ_MODEL):
        _llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0, timeout=30).bind_tools(
            [_SUBMIT_VERDICT_TOOL], tool_choice="auto", parallel_tool_calls=False
        )
        _llm_key_model = (GROQ_API_KEY, GROQ_MODEL)
    return _llm


def critic_node(state: GraphState) -> dict:
    """Returns: critic_verdict, critic_feedback, verdict_was_defaulted,
    total_steps, total_prompt_tokens, total_completion_tokens, cost_usd,
    step_log, and (only on an LLM-call failure) stopped_reason.
    """
    from agents.writer import _format_notes  # local import avoids a module-load cycle with writer.py

    prompt = (
        f"Question: {state['question']}\n\n"
        f"Researcher's notes:\n{_format_notes(state['research_notes'])}\n\n"
        f"Writer's draft:\n{state['draft_answer']}"
    )
    messages = [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=prompt)]

    total_steps = state["total_steps"] + 1
    step_log = list(state["step_log"])

    try:
        response = llm_module.call_llm(_get_llm(), messages)
    except APIStatusError as exc:
        status_code = getattr(exc, "status_code", None)
        stopped_reason = "rate_limited" if status_code == 429 else "malformed_tool_call"
        step_log.append(
            {
                "step": total_steps,
                "node": "critic",
                "tool": None,
                "arguments": None,
                "result_summary": f"LLM call failed: {status_code} {str(exc)[:150]}",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cumulative_cost_usd": round(state["cost_usd"], 6),
            }
        )
        return {"total_steps": total_steps, "step_log": step_log, "stopped_reason": stopped_reason}
    except APIConnectionError as exc:
        step_log.append(
            {
                "step": total_steps,
                "node": "critic",
                "tool": None,
                "arguments": None,
                "result_summary": f"LLM call failed: no response ({str(exc)[:150]})",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cumulative_cost_usd": round(state["cost_usd"], 6),
            }
        )
        return {"total_steps": total_steps, "step_log": step_log, "stopped_reason": "connection_error"}

    prompt_tokens, completion_tokens = llm_module.usage_from(response)
    total_prompt_tokens = state["total_prompt_tokens"] + prompt_tokens
    total_completion_tokens = state["total_completion_tokens"] + completion_tokens
    cost = llm_module.cost_usd(total_prompt_tokens, total_completion_tokens)

    tool_calls = response.tool_calls or []
    if tool_calls:
        args = tool_calls[0]["args"] or {}
        verdict = args.get("verdict", "").strip()
        feedback = args.get("feedback", "").strip()
        defaulted = False
        if verdict not in ("approved", "revise_writer"):
            # Model called the tool but didn't use one of the two allowed
            # values -- same fail-open reasoning as a missing tool call
            # below: don't loop retrying a deterministic (temperature=0)
            # malformed response, just proceed as approved.
            verdict = "approved"
            defaulted = True
    else:
        # No tool call at all, despite tool_choice="auto" -- fail open (see
        # module docstring): treat as approved, log the raw reply as
        # feedback, move on. The global step/revision caps already bound
        # any downstream damage from a wrongly-approved draft either way.
        verdict = "approved"
        feedback = response.content or "(critic did not call submit_verdict; defaulted to approved)"
        defaulted = True

    step_log.append(
        {
            "step": total_steps,
            "node": "critic",
            "tool": "submit_verdict" if tool_calls else None,
            "arguments": {"verdict": verdict, "feedback": feedback} if tool_calls else None,
            "result_summary": f"{verdict}: {feedback[:180]}",
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cumulative_cost_usd": round(cost, 6),
        }
    )

    return {
        "critic_verdict": verdict,
        "critic_feedback": feedback,
        "verdict_was_defaulted": defaulted,
        # Only incremented on an actual revise_writer verdict -- graph.py's
        # route_after_writer reads this to decide whether another
        # writer->critic round trip is still within MAX_REVISION_CYCLES.
        "revision_count": state["revision_count"] + (1 if verdict == "revise_writer" else 0),
        "total_steps": total_steps,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "cost_usd": cost,
        "step_log": step_log,
        "stopped_reason": None,
    }
