import numpy as np

from src.embedding_index import NumpyEmbeddingIndex
from src.retrieval_pipeline import RetrievalPipeline


class StaticBiEncoder:
    def encode_query(self, query):
        return np.array([1.0, 0.0], dtype=np.float32)


def test_retrieval_pipeline_returns_ranked_candidates(dataset):
    embeddings = np.zeros((len(dataset.documents), 2), dtype=np.float32)
    embeddings[:, 1] = 1.0
    embeddings[0] = np.array([1.0, 0.0])
    index = NumpyEmbeddingIndex(
        embeddings,
        dataset.documents["document_id"].astype(str).tolist(),
    )
    pipeline = RetrievalPipeline(dataset.documents, StaticBiEncoder(), index)
    result = pipeline.retrieve("quality complaints", top_k=3)

    assert len(result.candidates) == 3
    assert result.candidates.iloc[0]["document_id"] == "DOC001"
    assert result.candidates["retrieval_rank"].tolist() == [1, 2, 3]
