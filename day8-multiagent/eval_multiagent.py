"""Compares baseline/single_agent.py (a frozen duplicate of day4-rag's Day 7
agent) against agents/graph.py's multi-agent system, on identical questions
-- same spirit and structure as day4-rag/eval_agent.py, which compares
Day 7's agent against plain RAG.

Run: python eval_multiagent.py <single_agent|multiagent> [data/multiagent_eval_set.json]

Grading logic (_normalize/_answer_is_correct) is duplicated from
day4-rag/eval.py, not imported -- same "duplication over cross-repo
coupling" reasoning as agents/db.py. Simplified relative to the original:
this project's eval set has no "unanswerable" question type (day4-rag's
_is_refusal check doesn't apply here), so _answer_is_correct is just the
must_contain substring check.

Doesn't need its own cost-from-tokens formula the way eval.py did --
run_agent and run_multiagent both already return a real "cost_usd" from
their own internal accounting (see agents/llm.py's cost_usd, shared by
both baseline/single_agent.py and agents/*), so this script just reads
that field directly rather than recomputing it a third time.
"""

import json
import sys
import time
from collections import Counter

from agents.db import ping
from agents.graph import run_multiagent
from baseline.single_agent import DEFAULT_MAX_COST_USD, DEFAULT_MAX_STEPS, run_agent

# Single-agent pacing matches day4-rag/eval_agent.py's already-widened
# _INTER_QUESTION_DELAY_SECONDS_AGENT (45s, only reached after live 429
# storms on a tighter number -- see that module's comment). Multi-agent
# starts even wider (60s): a 3-node graph makes strictly more Groq calls
# per question than the single agent that 45s number was tuned for, so
# starting tighter would just repeat the same tuning-by-live-failure cycle
# documented across this whole curriculum's history.
_INTER_QUESTION_DELAY_SECONDS_SINGLE = 45
_INTER_QUESTION_DELAY_SECONDS_MULTIAGENT = 60

_PUNCTUATION_NORMALIZATION = str.maketrans(
    {
        "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-",  # hyphens/dashes
        "‘": "'", "’": "'",  # single curly quotes
        "“": '"', "”": '"',  # double curly quotes
    }
)


def _normalize(text: str) -> str:
    # gpt-oss-120b routinely wraps numbers in narrow no-break spaces
    # (U+202F, e.g. "67 million" instead of "67 million") as a
    # formatting habit -- found live: this silently failed a must_contain
    # check ("67 million" with a plain space) against an answer that was
    # actually correct, mis-grading a real pass as a fail. str.isspace()
    # covers U+202F, U+00A0 (no-break space), U+2009 (thin space), and
    # every other Unicode whitespace variant, not just the ASCII one --
    # collapsing all of them to a plain space (then re-collapsing runs)
    # makes the comparison robust to whichever one a given model prefers,
    # the same "verify before reporting" discipline that caught Day 5's and
    # Day 7's earlier must_contain phrasing mismatches.
    text = text.translate(_PUNCTUATION_NORMALIZATION)
    text = "".join(" " if ch.isspace() else ch for ch in text)
    text = " ".join(text.split())
    return text.lower()


def _answer_is_correct(entry: dict, answer: str) -> bool:
    normalized_answer = _normalize(answer)
    return all(_normalize(phrase) in normalized_answer for phrase in entry["must_contain"])


def _per_role_breakdown(step_log: list[dict]) -> dict:
    """Multi-agent-only: {"researcher": {"steps": n, "cost_usd": x}, ...} --
    the metric that explains WHERE any extra cost/latency over the single
    agent actually goes, not just that there is some.
    """
    breakdown: dict[str, dict] = {}
    prev_cost = 0.0
    for step in step_log:
        node = step["node"]
        entry = breakdown.setdefault(node, {"steps": 0, "cost_usd": 0.0})
        entry["steps"] += 1
        entry["cost_usd"] += max(0.0, step["cumulative_cost_usd"] - prev_cost)
        prev_cost = step["cumulative_cost_usd"]
    for entry in breakdown.values():
        entry["cost_usd"] = round(entry["cost_usd"], 6)
    return breakdown


def _critic_first_pass_approved(step_log: list[dict]) -> bool | None:
    """None if the critic never ran this question (a hard limit fired
    first) -- not counted as either an approval or a rejection."""
    for step in step_log:
        if step["node"] == "critic":
            return step["result_summary"].startswith("approved")
    return None


def run_comparison(eval_set_path: str, system: str, max_steps: int = DEFAULT_MAX_STEPS, max_cost_usd: float = DEFAULT_MAX_COST_USD) -> dict:
    assert system in ("single_agent", "multiagent"), f"unknown system {system!r}"

    with open(eval_set_path, encoding="utf-8") as f:
        eval_set = json.load(f)

    correct = 0
    total_cost_usd = 0.0
    query_seconds: list[float] = []
    step_counts: list[int] = []
    stopped_reasons: list[str] = []
    revision_counts: list[int] = []
    critic_first_pass: list[bool] = []
    per_question: list[dict] = []

    for i, entry in enumerate(eval_set):
        if i > 0:
            delay = _INTER_QUESTION_DELAY_SECONDS_SINGLE if system == "single_agent" else _INTER_QUESTION_DELAY_SECONDS_MULTIAGENT
            time.sleep(delay)

        start = time.perf_counter()
        if system == "single_agent":
            result = run_agent(entry["question"], max_steps=max_steps, max_cost_usd=max_cost_usd)
        else:
            result = run_multiagent(entry["question"])
        elapsed = time.perf_counter() - start

        answer = result["answer"]
        cost = result["cost_usd"]
        steps = len(result["steps"])
        stopped_reason = result["stopped_reason"]

        query_seconds.append(elapsed)
        total_cost_usd += cost
        step_counts.append(steps)
        stopped_reasons.append(stopped_reason)

        is_correct = _answer_is_correct(entry, answer)
        if is_correct:
            correct += 1

        question_record = {
            "id": entry["id"],
            "question": entry["question"],
            "correct": is_correct,
            "answer": answer,
            "steps": steps,
            "step_log": result["steps"],
            "stopped_reason": stopped_reason,
            "cost_usd": round(cost, 6),
            "seconds": round(elapsed, 2),
        }

        if system == "multiagent":
            revision_count = sum(
                1 for s in result["steps"] if s["node"] == "critic" and s["result_summary"].startswith("revise_writer")
            )
            revision_counts.append(revision_count)
            question_record["revision_count"] = revision_count
            question_record["per_role"] = _per_role_breakdown(result["steps"])
            first_pass = _critic_first_pass_approved(result["steps"])
            if first_pass is not None:
                critic_first_pass.append(first_pass)
            question_record["critic_first_pass_approved"] = first_pass

        per_question.append(question_record)
        print(
            f"Q{entry['id']:>2} [{'PASS' if is_correct else 'FAIL'}] "
            f"steps={steps} cost=${cost:.5f} {elapsed:.2f}s :: {entry['question'][:70]}"
        )

    total = len(eval_set)
    avg_seconds = sum(query_seconds) / len(query_seconds) if query_seconds else 0.0

    print()
    print(f"=== {system} on {eval_set_path} ===")
    print(f"Correct:  {correct}/{total}  ({round(100*correct/total)}%)")
    print(f"Cost:     ${total_cost_usd:.4f} total, ${total_cost_usd/total:.5f}/query")
    print(f"Latency:  {avg_seconds:.2f} sec/query (avg)")
    print(f"Steps:    {sum(step_counts)/len(step_counts):.1f} avg, max {max(step_counts)}")
    print(f"Stopped:  {dict(Counter(stopped_reasons))}")
    if system == "multiagent":
        print(f"Revisions: {sum(revision_counts)/len(revision_counts):.2f} avg per question")
        if critic_first_pass:
            approved_rate = sum(critic_first_pass) / len(critic_first_pass)
            print(f"Critic first-pass approval rate: {approved_rate:.0%} ({len(critic_first_pass)} questions reached the critic)")

    return {
        "system": system,
        "eval_set_path": eval_set_path,
        "total": total,
        "correct": correct,
        "cost_usd": total_cost_usd,
        "avg_seconds_per_query": avg_seconds,
        "avg_steps": sum(step_counts) / len(step_counts) if step_counts else 0,
        "max_steps_seen": max(step_counts) if step_counts else 0,
        "stopped_reasons": stopped_reasons,
        "per_question": per_question,
    }


if __name__ == "__main__":
    ping()  # fails fast if day4-rag's Chroma isn't reachable -- see agents/db.py
    system_arg = sys.argv[1] if len(sys.argv) > 1 else "multiagent"
    path_arg = sys.argv[2] if len(sys.argv) > 2 else "data/multiagent_eval_set.json"
    dump_full = len(sys.argv) > 3 and sys.argv[3] == "--full"
    output = run_comparison(path_arg, system_arg)

    if dump_full:
        print("\n=== FULL TRANSCRIPT (JSON) ===")
        print(json.dumps(output["per_question"], indent=2))
