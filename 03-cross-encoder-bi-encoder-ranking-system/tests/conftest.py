from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset_loader import load_ranking_dataset
from src.embedding_index import NumpyEmbeddingIndex
from src.settings import Settings


class FakeBiEncoder:
    model_name = "fake-bi-encoder"

    def __init__(self) -> None:
        self.document_vectors = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.8, 0.2, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    def encode(self, texts, **kwargs):
        count = len(texts)
        if count <= len(self.document_vectors):
            return self.document_vectors[:count]
        return np.vstack(
            [
                self.document_vectors,
                np.tile(self.document_vectors[-1], (count - len(self.document_vectors), 1)),
            ]
        )

    def encode_query(self, query):
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)


class FakeCrossEncoder:
    model_name = "fake-cross-encoder"

    def score(self, query, documents):
        # Deliberately reverses the first-stage preference.
        return np.arange(len(documents), dtype=np.float32)


@pytest.fixture
def settings():
    return Settings.from_yaml(PROJECT_ROOT / "config.yaml")


@pytest.fixture
def dataset(settings):
    return load_ranking_dataset(
        settings.documents_path,
        settings.queries_path,
        settings.qrels_path,
    )


@pytest.fixture
def ready_index(dataset):
    count = len(dataset.documents)
    embeddings = np.zeros((count, 3), dtype=np.float32)
    embeddings[:, 2] = 1.0
    embeddings[0] = np.array([1.0, 0.0, 0.0])
    embeddings[1] = np.array([0.8, 0.2, 0.0])
    return NumpyEmbeddingIndex(
        embeddings,
        dataset.documents["document_id"].astype(str).tolist(),
    )
