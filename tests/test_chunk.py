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


# --- Day 6: paragraph-aware packing -------------------------------------
# The Q7 finding (RESULTS.md): a short numbered list, chunked by pure word
# count, got its items scattered across chunks purely by where the Nth
# word fell -- not because any item was individually too long. These tests
# cover the fix: pack whole paragraphs together, never split one across a
# boundary unless it alone exceeds chunk_size.

def test_a_short_paragraph_is_never_split_across_two_chunks():
    # Two 8-word paragraphs, chunk_size=10 -- the second doesn't fit
    # alongside the first (8+8=16 > 10), so it must start a fresh chunk
    # whole, not have word 9 of it torn off into chunk 0.
    para_a = _words(8, prefix="a")
    para_b = _words(8, prefix="b")
    text = f"{para_a}\n\n{para_b}"

    chunks = chunk_text(text, chunk_size=10, overlap=2)

    assert para_a in chunks[0]
    assert para_b not in chunks[0]
    assert any(para_b in c for c in chunks[1:])


def test_several_short_paragraphs_pack_into_one_chunk_together():
    # Four ~7-word "numbered fact" style paragraphs, well under a 200-word
    # chunk_size -- exactly the grizzly-bear-article shape from Q7. All
    # four must land in the same chunk instead of one-per-chunk.
    facts = [f"{i}. Fact: " + _words(6, prefix=f"f{i}_") for i in range(1, 5)]
    text = "\n\n".join(facts)

    chunks = chunk_text(text, chunk_size=200, overlap=40)

    assert len(chunks) == 1
    assert all(fact in chunks[0] for fact in facts)


def test_a_paragraph_longer_than_chunk_size_still_falls_back_to_word_slicing():
    # One paragraph alone (40 words) exceeds chunk_size=10 -- must still
    # get split (there's no way to keep it whole), via the same
    # word-count slide the pre-Day-6 chunker always used.
    text = _words(40)

    chunks = chunk_text(text, chunk_size=10, overlap=2)

    assert len(chunks) > 1
    seen = set()
    for c in chunks:
        seen.update(c.split())
    assert seen == set(text.split())


def test_overlap_is_seeded_into_the_next_chunk_when_it_still_fits():
    para_a = _words(9, prefix="a")
    para_b = _words(9, prefix="b")
    text = f"{para_a}\n\n{para_b}"

    chunks = chunk_text(text, chunk_size=15, overlap=3)

    # chunk 0 = para_a (9 words); para_b (9 words) doesn't fit alongside it
    # (18 > 15), so chunk 1 opens with the last 3 words of chunk 0 -- and
    # 3 + 9 = 12 still fits within chunk_size=15, so the seed is kept.
    assert chunks[0].split() == para_a.split()
    assert chunks[1].split()[:3] == para_a.split()[-3:]
    assert para_b in chunks[1]
    assert len(chunks[1].split()) == 12


def test_overlap_is_dropped_rather_than_splitting_the_next_paragraph():
    # Same shape as above, but chunk_size=10 this time: the 3-word overlap
    # seed plus para_b's 9 words would be 12 > 10. Splitting para_b to make
    # room would defeat the whole point of paragraph-aware chunking, so
    # the overlap is dropped for this one boundary instead -- chunk 1 is
    # para_b alone, whole, un-split.
    para_a = _words(9, prefix="a")
    para_b = _words(9, prefix="b")
    text = f"{para_a}\n\n{para_b}"

    chunks = chunk_text(text, chunk_size=10, overlap=3)

    assert chunks[1].split() == para_b.split()


def test_blank_lines_with_only_whitespace_are_not_treated_as_paragraphs():
    text = "first paragraph\n\n   \n\nsecond paragraph"
    chunks = chunk_text(text, chunk_size=200, overlap=10)
    assert chunks == ["first paragraph second paragraph"]


def test_a_markdown_header_and_its_own_paragraphs_stay_together():
    # The actual Q7 shape: a header followed by TWO of its own paragraphs
    # (blank-line separated) before the next header. Both paragraphs must
    # land in the same chunk as their header, not just the header's
    # immediate next line.
    section_1 = "#### 1. Fact: " + _words(5, "s1a") + "\n\n" + _words(5, "s1b") + "\n\n" + _words(5, "s1c")
    section_2 = "#### 2. Fact: " + _words(5, "s2a")
    text = section_1 + "\n\n" + section_2

    chunks = chunk_text(text, chunk_size=200, overlap=10)

    assert len(chunks) == 1
    assert "s1a0" in chunks[0] and "s1b0" in chunks[0] and "s1c0" in chunks[0] and "s2a0" in chunks[0]


def test_a_header_section_is_not_split_from_its_own_content_even_across_chunks():
    # Two header sections, each big enough that both together don't fit
    # in one chunk -- section 2 must still open its own chunk whole,
    # never torn apart from its own paragraph.
    section_1 = "#### 1. Fact: " + _words(15, "s1")
    section_2 = "#### 2. Fact: " + _words(15, "s2")
    text = section_1 + "\n\n" + section_2

    chunks = chunk_text(text, chunk_size=18, overlap=2)

    section_2_chunks = [c for c in chunks if "s2" in c]
    assert len(section_2_chunks) >= 1
    assert all(f"s2{i}" in section_2_chunks[0] for i in range(15))


def test_text_with_no_markdown_headers_still_uses_plain_paragraph_splitting():
    # No "#"-prefixed lines anywhere -- must fall back to the ordinary
    # blank-line paragraph behavior, not try to treat the whole text as
    # one unsplittable section.
    text = "plain paragraph one\n\nplain paragraph two"
    chunks = chunk_text(text, chunk_size=3, overlap=1)
    assert len(chunks) > 1
