from src.text_preprocessing import clean_text, split_sentences, validate_article, word_count


def test_clean_text_preserves_unicode_and_facts() -> None:
    text = "<p>José reported 24% improvement on 12 July 2026.</p>\n\n"
    cleaned = clean_text(text)
    assert cleaned == "José reported 24% improvement on 12 July 2026."


def test_sentence_split_and_word_count() -> None:
    sentences = split_sentences("First sentence. Second sentence! Third?")
    assert len(sentences) == 3
    assert word_count("Quality improved by 24 percent.") == 5


def test_validate_article_rejects_short_text() -> None:
    try:
        validate_article("Too short", min_words=3)
    except ValueError as exc:
        assert "at least 3 words" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
