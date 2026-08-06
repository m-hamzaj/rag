"""CLI entrypoint.

`python -m rag.cli "question"` answers once and exits.
`python -m rag.cli` with no arguments starts an interactive loop.
"""

import sys

from rag.ask import ask


def _print_result(result: dict) -> None:
    print(f"\n{result['answer']}")
    if result["citations"]:
        print("\nSources:")
        for c in result["citations"]:
            print(f"  - {c['title']} ({c['url']})")


def _run_once(question: str) -> None:
    _print_result(ask(question))


def _run_interactive() -> None:
    print("Day 4 RAG -- ask a question about the scraped articles ('exit' or Ctrl+C to quit).")
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
    if len(sys.argv) > 1:
        _run_once(" ".join(sys.argv[1:]))
    else:
        _run_interactive()


if __name__ == "__main__":
    main()
