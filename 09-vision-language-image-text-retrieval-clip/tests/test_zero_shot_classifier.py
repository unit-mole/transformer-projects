import numpy as np
from src.zero_shot_classifier import classify_from_embeddings, softmax


def test_softmax_sums_to_one():
    assert np.isclose(softmax(np.array([0.1, 0.2, 0.3])).sum(), 1.0)


def test_classification_ranks_best_label():
    predictions = classify_from_embeddings(
        np.array([1.0, 0.0]),
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        ["car", "cat"],
    )
    assert predictions[0].label == "car"
