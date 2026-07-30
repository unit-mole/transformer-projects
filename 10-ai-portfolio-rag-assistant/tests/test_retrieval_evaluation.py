import numpy as np
from src.retrieval_evaluation import rank

def test_rank_returns_closest_project():
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    result = rank(np.array([0.9, 0.1], dtype=np.float32), matrix, ["a", "b"], 1)
    assert result == ["a"]
