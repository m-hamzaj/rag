import sys

from rag import cli as cli_module


def test_print_result_shows_answer_and_sources(capsys):
    cli_module._print_result({
        "answer": "It is X.",
        "citations": [{"title": "Article One", "url": "https://x/1"}],
    })
    out = capsys.readouterr().out
    assert "It is X." in out
    assert "Sources:" in out
    assert "Article One (https://x/1)" in out


def test_print_result_omits_sources_section_when_no_citations(capsys):
    cli_module._print_result({"answer": "I don't know.", "citations": []})
    out = capsys.readouterr().out
    assert "I don't know." in out
    assert "Sources:" not in out


def test_run_once_calls_ask_with_the_question(monkeypatch, capsys):
    captured = {}

    def fake_ask(q):
        captured["q"] = q
        return {"answer": "ok", "citations": []}

    monkeypatch.setattr(cli_module, "ask", fake_ask)

    cli_module._run_once("what eats jaguarundis?")

    assert captured["q"] == "what eats jaguarundis?"
    assert "ok" in capsys.readouterr().out


def test_main_with_argv_runs_once_and_joins_multiword_question(monkeypatch):
    captured = {}
    monkeypatch.setattr(cli_module, "_run_once", lambda q: captured.setdefault("q", q))
    monkeypatch.setattr(sys, "argv", ["cli.py", "what", "eats", "kale?"])

    cli_module.main()

    assert captured["q"] == "what eats kale?"


def test_main_without_argv_runs_interactive(monkeypatch):
    called = []
    monkeypatch.setattr(cli_module, "_run_interactive", lambda: called.append(True))
    monkeypatch.setattr(sys, "argv", ["cli.py"])

    cli_module.main()

    assert called == [True]


def test_run_interactive_exits_on_exit_command(monkeypatch, capsys):
    inputs = iter(["exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    called = []
    monkeypatch.setattr(cli_module, "_run_once", lambda q: called.append(q))

    cli_module._run_interactive()

    assert called == []  # never asked a real question


def test_run_interactive_skips_blank_input_and_asks_real_questions(monkeypatch):
    inputs = iter(["", "a real question", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    called = []
    monkeypatch.setattr(cli_module, "_run_once", lambda q: called.append(q))

    cli_module._run_interactive()

    assert called == ["a real question"]


def test_run_interactive_exits_cleanly_on_eof(monkeypatch):
    def raise_eof(prompt=""):
        raise EOFError

    monkeypatch.setattr("builtins.input", raise_eof)
    called = []
    monkeypatch.setattr(cli_module, "_run_once", lambda q: called.append(q))

    cli_module._run_interactive()  # must not raise

    assert called == []
