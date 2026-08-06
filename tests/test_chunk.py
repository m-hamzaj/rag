import pytest

from rag.chunk import chunk_text


def _words(n, prefix="word"):
    return " ".join(f"{prefix}{i}" for i in range(n))


def test_empty_text_returns_no_chunks():
    assert chunk_text("", chunk_size=10, overlap=2) == []


def test_whitespace_only_text_returns_no_chunks():
    assert chunk_text("   \n\t  ", chunk_size=10, overlap=2) == []


def test_text_shorter_than_chunk_size_returns_one_chunk():
    text = _words(5)
    chunks = chunk_text(text, chunk_size=10, overlap=2)
    assert chunks == [text]


def test_text_exactly_chunk_size_returns_one_chunk():
    text = _words(10)
    chunks = chunk_text(text, chunk_size=10, overlap=2)
    assert chunks == [text]


def test_chunks_advance_by_chunk_size_minus_overlap():
    text = _words(25)
    chunks = chunk_text(text, chunk_size=10, overlap=2)
    # step = 8: starts at word 0, 8, 16, 24
    assert chunks[0] == _words(10)
    assert chunks[1] == " ".join(f"word{i}" for i in range(8, 18))
    assert chunks[2] == " ".join(f"word{i}" for i in range(16, 25))


def test_consecutive_chunks_actually_overlap():
    text = _words(25)
    chunks = chunk_text(text, chunk_size=10, overlap=2)
    first_words = chunks[0].split()
    second_words = chunks[1].split()
    assert first_words[-2:] == second_words[:2]


def test_last_chunk_is_short_rather_than_padded():
    text = _words(22)
    chunks = chunk_text(text, chunk_size=10, overlap=2)
    assert len(chunks[-1].split()) < 10
    assert chunks[-1] == " ".join(f"word{i}" for i in range(16, 22))


def test_no_word_is_dropped_across_chunk_boundaries():
    text = _words(37)
    chunks = chunk_text(text, chunk_size=10, overlap=3)
    seen = set()
    for c in chunks:
        seen.update(c.split())
    assert seen == set(text.split())


def test_overlap_equal_to_chunk_size_raises():
    with pytest.raises(ValueError):
        chunk_text(_words(20), chunk_size=10, overlap=10)


def test_overlap_greater_than_chunk_size_raises():
    with pytest.raises(ValueError):
        chunk_text(_words(20), chunk_size=10, overlap=15)


def test_zero_overlap_is_allowed():
    text = _words(20)
    chunks = chunk_text(text, chunk_size=10, overlap=0)
    assert chunks == [_words(10), " ".join(f"word{i}" for i in range(10, 20))]


def test_default_config_values_are_used_when_not_passed():
    # chunk_size/overlap default to rag.config's CHUNK_SIZE_WORDS/CHUNK_OVERLAP_WORDS
    text = _words(500)
    chunks = chunk_text(text)
    assert len(chunks) > 1
