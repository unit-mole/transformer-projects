import numpy as np
from src.inference_pipeline import get_top_k_predictions, softmax


def test_softmax():
    result = softmax(np.array([1.0, 2.0, 3.0]))
    assert np.isclose(result.sum(), 1.0)


def test_top_k():
    result = get_top_k_predictions(np.array([0.1, 0.7, 0.2]), {0:"a",1:"b",2:"c"}, 2)
    assert [row["label"] for row in result] == ["b", "c"]
