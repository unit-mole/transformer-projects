from src.text_preprocessing import clean_text, combine_title_and_document


def test_clean_text_preserves_unicode_and_domain_terms():
    raw = "  <b>MiniLM</b>\u00a0supports  café  and  ORP-12  "
    cleaned = clean_text(raw)
    assert cleaned == "MiniLM supports café and ORP-12"


def test_combine_title_and_document():
    assert combine_title_and_document("Ranking", "Fast retrieval") == (
        "Ranking. Fast retrieval"
    )
