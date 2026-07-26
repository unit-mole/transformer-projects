import numpy as np

from src.embedding_index import NumpyEmbeddingIndex
from src.ranking_engine import TwoStageRankingEngine


class StaticBiEncoder:
    model_name = "static-bi"

    def encode(self, texts, **kwargs):
        matrix = np.zeros((len(texts), 2), dtype=np.float32)
        matrix[:, 1] = 1.0
        if len(texts):
            matrix[0] = np.array([1.0, 0.0])
        return matrix

    def encode_query(self, query):
        return np.array([1.0, 0.0], dtype=np.float32)


class StaticCrossEncoder:
    model_name = "static-cross"

    def score(self, query, documents):
        return np.linspace(0.0, 1.0, num=len(documents), dtype=np.float32)


def test_two_stage_engine_search(dataset, settings):
    embeddings = StaticBiEncoder().encode(dataset.documents["search_text"].tolist())
    index = NumpyEmbeddingIndex(
        embeddings,
        dataset.documents["document_id"].astype(str).tolist(),
    )
    engine = TwoStageRankingEngine(
        dataset=dataset,
        settings=settings,
        bi_encoder=StaticBiEncoder(),
        cross_encoder=StaticCrossEncoder(),
        index=index,
    )
    response = engine.search(
        "find similar quality complaints",
        candidate_k=5,
        rerank_k=5,
    )

    assert len(response.candidates) == 5
    assert len(response.reranked_results) == 5
    assert "cross_encoder_score" in response.reranked_results.columns
    assert response.latency.total_search_ms >= 0
