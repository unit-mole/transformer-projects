import numpy as np
from src.retrieval_engine import cosine_scores, rank_embeddings


def test_cosine_and_ranking():
    gallery = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])
    scores = cosine_scores(np.array([1.0, 0.0]), gallery)
    assert scores[0] > scores[2] > scores[1]
    results = rank_embeddings(np.array([1.0, 0.0]), gallery, ["x", "y", "z"], top_k=2)
    assert [item.image_id for item in results] == ["x", "z"]
