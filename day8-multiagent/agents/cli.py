"""CLI entrypoint.

`python -m agents.cli "question"` answers once and exits.
`python -m agents.cli` with no arguments starts an interactive loop.
"""

import sys

from agents.db import ping
from agents.graph import run_multiagent


def _print_result(result: dict) -> None:
    print(f"\n{result['answer']}")
    if result["citations"]:
        print("\nSources:")
        for c in result["citations"]:
            print(f"  - {c['title']} ({c['url']})")
    print(f"\n[stopped_reason={result['stopped_reason']}  cost=${result['cost_usd']:.4f}  steps={len(result['steps'])}]")


def _run_once(question: str) -> None:
    _print_result(run_multiagent(question))


def _run_interactive() -> None:
    print("Day 8 multi-agent -- ask a question about the scraped articles ('exit' or Ctrl+C to quit).")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            break
        _run_once(question)


def main() -> None:
    # Fails fast and clearly if day4-rag's Chroma isn't reachable, rather
    # than surfacing an opaque error three LLM calls deep inside the
    # researcher's first tool call -- see agents/db.py's ping() docstring.
    ping()
    if len(sys.argv) > 1:
        _run_once(" ".join(sys.argv[1:]))
    else:
        _run_interactive()


if __name__ == "__main__":
    main()
