import numpy as np
import pandas as pd

from src.reranking_pipeline import RerankingPipeline


class ReverseCrossEncoder:
    def score(self, query, documents):
        return np.arange(len(documents), dtype=np.float32)


def test_reranking_records_rank_movement():
    candidates = pd.DataFrame(
        {
            "document_id": ["A", "B", "C"],
            "search_text": ["alpha", "beta", "gamma"],
            "retrieval_rank": [1, 2, 3],
            "bi_encoder_score": [0.9, 0.8, 0.7],
        }
    )
    result = RerankingPipeline(ReverseCrossEncoder()).rerank(
        "query",
        candidates,
        rerank_k=3,
    )

    assert result.results["document_id"].tolist() == ["C", "B", "A"]
    assert result.results.iloc[0]["rank_movement"] == 2
    assert result.reranking_ms >= 0
