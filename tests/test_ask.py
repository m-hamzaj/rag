from rag import ask as ask_module

_CHUNK_A = {"document_url": "https://x/1", "document_title": "Article One", "chunk_index": 0, "text": "t", "similarity": 0.9}
_CHUNK_B = {"document_url": "https://x/2", "document_title": "Article Two", "chunk_index": 0, "text": "t", "similarity": 0.8}
_CHUNK_RELATED = {"document_url": "https://x/3", "document_title": "Article Three", "chunk_index": 0, "text": "t", "similarity": 0.25}


def _retrieval(accepted=(), related=()):
    return {"accepted": list(accepted), "related": list(related)}


def test_ask_refuses_without_calling_the_llm_when_nothing_retrieved_at_all(monkeypatch):
    monkeypatch.setattr(ask_module, "retrieve", lambda q: _retrieval())
    called = []
    monkeypatch.setattr(ask_module, "generate_answer", lambda q, chunks, **kw: called.append(True))

    result = ask_module.ask("something not in the corpus")

    assert result == {"answer": ask_module.NO_ANSWER, "citations": []}
    assert called == []  # the LLM must never be called for a fast refusal


def test_ask_returns_answer_with_deduped_citations(monkeypatch):
    monkeypatch.setattr(ask_module, "retrieve", lambda q: _retrieval(accepted=[_CHUNK_A, _CHUNK_B]))
    monkeypatch.setattr(
        ask_module, "generate_answer",
        lambda q, chunks, **kw: {"answer": "The answer is X [1][2].", "citations": [_CHUNK_A, _CHUNK_B]},
    )

    result = ask_module.ask("a real question")

    assert result["answer"] == "The answer is X [1][2]."
    assert result["citations"] == [
        {"title": "Article One", "url": "https://x/1"},
        {"title": "Article Two", "url": "https://x/2"},
    ]


def test_ask_dedupes_citations_from_the_same_article(monkeypatch):
    chunk_a2 = {**_CHUNK_A, "chunk_index": 1, "text": "different chunk, same article"}
    monkeypatch.setattr(ask_module, "retrieve", lambda q: _retrieval(accepted=[_CHUNK_A, chunk_a2]))
    monkeypatch.setattr(
        ask_module, "generate_answer",
        lambda q, chunks, **kw: {"answer": "X [1][2].", "citations": [_CHUNK_A, chunk_a2]},
    )

    result = ask_module.ask("a real question")

    assert result["citations"] == [{"title": "Article One", "url": "https://x/1"}]


def test_ask_treats_llm_refusal_as_no_citations_even_with_retrieved_chunks(monkeypatch):
    # Retrieval found plausible chunks, but the LLM itself decided they
    # don't actually answer the question -- citations must not leak
    # through for a refusal the model produced on its own.
    monkeypatch.setattr(ask_module, "retrieve", lambda q: _retrieval(accepted=[_CHUNK_A]))
    monkeypatch.setattr(
        ask_module, "generate_answer",
        lambda q, chunks, **kw: {"answer": "I don't know.", "citations": [_CHUNK_A]},
    )

    result = ask_module.ask("a question the chunks don't actually answer")

    assert result == {"answer": "I don't know.", "citations": []}


def test_ask_recognizes_refusal_variants_case_and_punctuation_insensitively(monkeypatch):
    monkeypatch.setattr(ask_module, "retrieve", lambda q: _retrieval(accepted=[_CHUNK_A]))
    for variant in ["I don't know", "i don't know.", "I DO NOT KNOW."]:
        monkeypatch.setattr(
            ask_module, "generate_answer",
            lambda q, chunks, variant=variant, **kw: {"answer": variant, "citations": [_CHUNK_A]},
        )
        result = ask_module.ask("q")
        assert result["citations"] == []


# --- The related tier: no accepted chunks, but something topically close.

def test_ask_falls_back_to_related_tier_when_nothing_is_accepted(monkeypatch):
    monkeypatch.setattr(ask_module, "retrieve", lambda q: _retrieval(related=[_CHUNK_RELATED]))
    captured = {}

    def fake_generate(q, chunks, **kw):
        captured["chunks"] = chunks
        captured["related_kwarg"] = kw.get("related")
        return {"answer": "The sources don't directly cover this, but generally...", "citations": [_CHUNK_RELATED]}

    monkeypatch.setattr(ask_module, "generate_answer", fake_generate)

    result = ask_module.ask("something only tangentially covered")

    assert captured["chunks"] == [_CHUNK_RELATED]
    assert captured["related_kwarg"] is True  # the caveated prompt, not the direct one
    assert result["answer"].startswith(ask_module._RELATED_PREFIX)
    assert "generally..." in result["answer"]
    assert result["citations"] == [{"title": "Article Three", "url": "https://x/3"}]


def test_ask_related_tier_still_refuses_if_the_llm_finds_nothing_useful(monkeypatch):
    monkeypatch.setattr(ask_module, "retrieve", lambda q: _retrieval(related=[_CHUNK_RELATED]))
    monkeypatch.setattr(
        ask_module, "generate_answer",
        lambda q, chunks, **kw: {"answer": "I don't know.", "citations": [_CHUNK_RELATED]},
    )

    result = ask_module.ask("q")

    # A refusal from the related tier must look exactly like any other
    # refusal -- no stray prefix, no leaked citations.
    assert result == {"answer": "I don't know.", "citations": []}


def test_ask_prefers_accepted_over_related_when_both_are_present(monkeypatch):
    monkeypatch.setattr(
        ask_module, "retrieve", lambda q: _retrieval(accepted=[_CHUNK_A], related=[_CHUNK_RELATED])
    )
    captured = {}

    def fake_generate(q, chunks, **kw):
        captured["chunks"] = chunks
        captured["related_kwarg"] = kw.get("related")
        return {"answer": "Direct answer [1].", "citations": [_CHUNK_A]}

    monkeypatch.setattr(ask_module, "generate_answer", fake_generate)

    result = ask_module.ask("q")

    assert captured["chunks"] == [_CHUNK_A]  # related chunks never even passed to generate_answer
    assert captured["related_kwarg"] in (None, False)
    assert result["answer"] == "Direct answer [1]."  # no related prefix
