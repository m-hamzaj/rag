"""Day 7 -- runs either plain RAG (rag/ask.py) or the tool-calling agent
(rag/agent.py) against a question set, grading both the same way eval.py
already does (must_contain substring check, see that module's docstring
for the full reasoning), so the two systems are comparable on identical
terms.

Run: python eval_agent.py <plain|agent> <data/eval_set.json|data/agent_eval_set.json>

Reuses eval.py's grading/normalization/pricing helpers rather than
duplicating them -- the whole point of this script is a fair, identical
comparison, which a second copy of the same logic could quietly drift
from.
"""

import json
import sys
import time

import httpx

from eval import _answer_is_correct, _run_cost_usd
from rag.agent import DEFAULT_MAX_COST_USD, DEFAULT_MAX_STEPS, run_agent
from rag.ask import ask

# Same reasoning as eval.py's own constants: separate real Groq calls
# (each question here can be several calls deep for the agent) need
# pacing and retry room, or a transient 429 kills the whole batch.
_RATE_LIMIT_RETRIES = 4
_RATE_LIMIT_BASE_BACKOFF_SECONDS = 15
_INTER_QUESTION_DELAY_SECONDS = 5

# Measured live: an agent question with a large, growing message history
# (full article reads carried forward every step) can leave enough of the
# per-minute token budget consumed that even rag/agent.py's own inter-step
# pacing and retries weren't reliably enough -- two consecutive agent
# questions each burned ~10 minutes on sustained 429s before this was
# widened. 45s between agent questions specifically, plain RAG's single
# small request per question doesn't need nearly this much room.
_INTER_QUESTION_DELAY_SECONDS_AGENT = 45


def _ask_plain_with_retry(question: str) -> dict:
    for attempt in range(_RATE_LIMIT_RETRIES + 1):
        try:
            return ask(question)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 429 or attempt == _RATE_LIMIT_RETRIES:
                raise
            time.sleep(_RATE_LIMIT_BASE_BACKOFF_SECONDS * (2**attempt))
    raise AssertionError("unreachable")


def run_comparison(eval_set_path: str, system: str, max_steps: int = DEFAULT_MAX_STEPS, max_cost_usd: float = DEFAULT_MAX_COST_USD) -> dict:
    assert system in ("plain", "agent"), f"unknown system {system!r}"

    with open(eval_set_path, encoding="utf-8") as f:
        eval_set = json.load(f)

    correct = 0
    total_prompt_tokens = total_completion_tokens = 0
    total_cost_usd = 0.0
    query_seconds: list[float] = []
    step_counts: list[int] = []
    stopped_reasons: list[str] = []
    per_question: list[dict] = []

    for i, entry in enumerate(eval_set):
        if i > 0:
            time.sleep(_INTER_QUESTION_DELAY_SECONDS if system == "plain" else _INTER_QUESTION_DELAY_SECONDS_AGENT)

        start = time.perf_counter()
        step_log = None
        if system == "plain":
            result = _ask_plain_with_retry(entry["question"])
            answer = result["answer"]
            cost = _run_cost_usd(result["usage"]["prompt_tokens"], result["usage"]["completion_tokens"])
            steps = 1
            stopped_reason = "n/a"
        else:
            result = run_agent(entry["question"], max_steps=max_steps, max_cost_usd=max_cost_usd)
            answer = result["answer"]
            cost = result["cost_usd"]
            steps = len(result["steps"])
            stopped_reason = result["stopped_reason"]
            step_log = result["steps"]
        elapsed = time.perf_counter() - start

        query_seconds.append(elapsed)
        total_prompt_tokens += result["usage"]["prompt_tokens"]
        total_completion_tokens += result["usage"]["completion_tokens"]
        total_cost_usd += cost
        step_counts.append(steps)
        stopped_reasons.append(stopped_reason)

        is_correct = _answer_is_correct(entry, answer)
        if is_correct:
            correct += 1

        per_question.append(
            {
                "id": entry["id"],
                "question": entry["question"],
                "correct": is_correct,
                "answer": answer,
                "steps": steps,
                "step_log": step_log,
                "stopped_reason": stopped_reason,
                "cost_usd": round(cost, 6),
                "seconds": round(elapsed, 2),
            }
        )
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
    if system == "agent":
        print(f"Steps:    {sum(step_counts)/len(step_counts):.1f} avg, max {max(step_counts)}")
        from collections import Counter

        print(f"Stopped:  {dict(Counter(stopped_reasons))}")

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
    system_arg = sys.argv[1] if len(sys.argv) > 1 else "agent"
    path_arg = sys.argv[2] if len(sys.argv) > 2 else "data/agent_eval_set.json"
    max_steps_arg = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_MAX_STEPS
    dump_full = len(sys.argv) > 4 and sys.argv[4] == "--full"
    output = run_comparison(path_arg, system_arg, max_steps=max_steps_arg)

    if dump_full:
        print("\n=== FULL TRANSCRIPT (JSON) ===")
        print(json.dumps(output["per_question"], indent=2))
