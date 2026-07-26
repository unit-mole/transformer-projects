from src.text_preprocessing import clean_markdown, extract_sections


def test_clean_markdown_preserves_code_and_model_names():
    text = "## Model\n```python\nmodel = 'all-MiniLM-L6-v2'\n```"
    cleaned = clean_markdown(text)
    assert "all-MiniLM-L6-v2" in cleaned
    assert "```" not in cleaned


def test_extract_sections_uses_headings():
    sections = extract_sections("# Title\nIntro text.\n## Metrics\nRecall@K and MRR.")
    assert [section.title for section in sections] == ["Title", "Metrics"]
