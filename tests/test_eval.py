"""Tests for eval.py's own grading logic -- separate from rag/'s test
suite since eval.py lives at the project root, not in the rag package.
"""

from eval import _answer_is_correct, _normalize, _run_cost_usd, _top_ranked_articles


def test_normalize_maps_unicode_hyphens_to_ascii():
    # Found live: openai/gpt-oss-120b wrote "paddle‑shaped" (non-breaking
    # hyphen) where the eval set's must_contain used a plain ASCII one.
    assert _normalize("paddle‑shaped") == "paddle-shaped"
    assert _normalize("en–dash and em—dash") == "en-dash and em-dash"


def test_normalize_maps_curly_quotes_to_straight():
    assert _normalize("it’s a “test”") == "it's a \"test\""


def test_normalize_maps_vulgar_fractions_to_ascii():
    # Found live: "¼ cup" (a single Unicode glyph) never matched a
    # must_contain of "1/4 cup".
    assert _normalize("¼ cup of sugar") == "1/4 cup of sugar"
    assert _normalize("½ and ¾") == "1/2 and 3/4"


def test_normalize_maps_non_breaking_space_to_regular_space():
    assert _normalize("a b") == "a b"


def test_normalize_lowercases():
    assert _normalize("PADDLE-Shaped") == "paddle-shaped"


def test_answer_is_correct_checks_refusal_for_unanswerable_type():
    entry = {"type": "unanswerable", "must_contain": []}
    assert _answer_is_correct(entry, "I don't know.") is True
    assert _answer_is_correct(entry, "Here is a real answer.") is False


def test_answer_is_correct_checks_must_contain_for_other_types():
    entry = {"type": "single", "must_contain": ["Kiwa"]}
    assert _answer_is_correct(entry, "It is a Kiwa crab.") is True
    assert _answer_is_correct(entry, "It is a different crab.") is False


def test_answer_is_correct_requires_every_must_contain_phrase():
    entry = {"type": "multi", "must_contain": ["1939", "remote cameras"]}
    assert _answer_is_correct(entry, "Counted in 1939 using remote cameras.") is True
    assert _answer_is_correct(entry, "Counted in 1939 only.") is False


def test_answer_is_correct_normalizes_both_the_answer_and_must_contain():
    # The fix must work regardless of which side (the real answer, or the
    # hand-written must_contain phrase) happens to use the ASCII form.
    entry = {"type": "single", "must_contain": ["paddle-shaped"]}
    assert _answer_is_correct(entry, "A paddle‑shaped tail.") is True


def test_run_cost_usd_uses_real_groq_rates():
    # 1M prompt tokens @ $0.15 + 1M completion tokens @ $0.60, confirmed
    # against console.groq.com/docs/models for openai/gpt-oss-120b.
    assert _run_cost_usd(1_000_000, 1_000_000) == 0.75


def test_run_cost_usd_zero_tokens_is_zero_cost():
    assert _run_cost_usd(0, 0) == 0.0


def test_top_ranked_articles_dedupes_multiple_chunks_from_one_article(monkeypatch):
    import eval as eval_module

    chunks = [
        {"document_url": "https://x/a", "similarity": 0.9},
        {"document_url": "https://x/a", "similarity": 0.8},  # same article again
        {"document_url": "https://x/b", "similarity": 0.7},
    ]
    monkeypatch.setattr(eval_module, "embed_query", lambda q: [0.0])
    monkeypatch.setattr(eval_module, "search_similar_chunks", lambda emb, k: chunks)

    result = _top_ranked_articles("q", limit=5)

    assert result == ["https://x/a", "https://x/b"]


def test_top_ranked_articles_respects_the_limit(monkeypatch):
    import eval as eval_module

    chunks = [{"document_url": f"https://x/{i}", "similarity": 1.0 - i * 0.01} for i in range(10)]
    monkeypatch.setattr(eval_module, "embed_query", lambda q: [0.0])
    monkeypatch.setattr(eval_module, "search_similar_chunks", lambda emb, k: chunks)

    result = _top_ranked_articles("q", limit=3)

    assert result == ["https://x/0", "https://x/1", "https://x/2"]
