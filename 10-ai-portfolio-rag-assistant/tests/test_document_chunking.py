from src.document_chunking import chunk_markdown

def test_chunking_preserves_section_title():
    chunks = chunk_markdown("doc", "# Overview\n" + "word " * 300, size_words=100, overlap_words=20)
    assert len(chunks) > 1
    assert all(chunk.section == "Overview" for chunk in chunks)
