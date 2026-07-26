from pathlib import Path

from src.document_loader import load_document, load_documents


def test_load_document_parses_frontmatter(tmp_path: Path):
    path = tmp_path / "sample.md"
    path.write_text("""---
project_name: Demo Search
project_category: NLP
tags: semantic-search, sentence-bert
---
# Demo Search

Useful content.
""", encoding="utf-8")
    document = load_document(path, root=tmp_path)
    assert document.project_name == "Demo Search"
    assert document.project_category == "NLP"
    assert document.tags == ["semantic-search", "sentence-bert"]
    assert document.document_id == "sample"


def test_load_documents_skips_duplicate_content(tmp_path: Path):
    for name in ("one.md", "two.md"):
        (tmp_path / name).write_text("# Same\n\nDuplicate body.", encoding="utf-8")
    documents = load_documents(tmp_path)
    assert len(documents) == 1
