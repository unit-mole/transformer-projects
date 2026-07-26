import numpy as np

from src.embedding_index import NumpyEmbeddingIndex


def test_numpy_index_returns_highest_cosine_similarity():
    index = NumpyEmbeddingIndex(
        embeddings=np.array(
            [[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]],
            dtype=np.float32,
        ),
        document_ids=["A", "B", "C"],
    )
    indices, scores = index.search(np.array([1.0, 0.0]), top_k=2)
    ids = [index.document_ids[position] for position in indices]

    assert ids == ["A", "C"]
    assert scores[0] >= scores[1]


def test_index_round_trip(tmp_path):
    index = NumpyEmbeddingIndex(
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        ["A", "B"],
    )
    index.save(tmp_path, metadata={"bi_encoder_model": "test"})
    loaded, metadata = NumpyEmbeddingIndex.load(tmp_path)

    assert loaded.document_ids == ["A", "B"]
    assert loaded.embeddings.shape == (2, 2)
    assert metadata["bi_encoder_model"] == "test"
