from src.text_preprocessing import clean_markdown

def test_removes_frontmatter_but_keeps_heading():
    text = "---\ntitle: Test\n---\n# Heading\n\nMiniLM model"
    cleaned = clean_markdown(text)
    assert "title: Test" not in cleaned
    assert "# Heading" in cleaned
    assert "MiniLM" in cleaned
