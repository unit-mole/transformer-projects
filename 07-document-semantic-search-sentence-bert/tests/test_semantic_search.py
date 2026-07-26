import numpy as np

from src.semantic_search import cosine_similarity_matrix, search_with_vector


def test_cosine_similarity_ranks_expected_vector_first():
    chunks = [
        {"chunk_id": "a", "project_category": "NLP", "document_type": "readme"},
        {"chunk_id": "b", "project_category": "CV", "document_type": "readme"},
    ]
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    results = search_with_vector(np.array([0.9, 0.1]), chunks, matrix, top_k=2)
    assert results[0]["chunk_id"] == "a"
    assert results[0]["rank"] == 1


def test_cosine_similarity_handles_zero_safely():
    scores = cosine_similarity_matrix(np.array([0.0, 0.0]), np.array([[1.0, 0.0]]))
    assert float(scores[0]) == 0.0
