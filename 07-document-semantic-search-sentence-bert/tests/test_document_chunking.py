from src.document_chunking import split_text


def test_split_text_respects_size_and_overlap():
    text = " ".join(f"word{i}." for i in range(80))
    chunks = split_text(text, chunk_size_words=30, overlap_words=5)
    assert len(chunks) >= 3
    assert all(len(chunk.split()) <= 35 for chunk in chunks)


def test_split_text_rejects_invalid_overlap():
    try:
        split_text("text", chunk_size_words=10, overlap_words=10)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")
