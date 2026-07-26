from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np


class NumpyEmbeddingIndex:
    """Portable cosine-similarity index for small CPU-based demos."""

    EMBEDDINGS_FILE = "embeddings.npy"
    IDS_FILE = "document_ids.json"
    METADATA_FILE = "index_metadata.json"

    def __init__(
        self,
        embeddings: np.ndarray | None = None,
        document_ids: Sequence[str] | None = None,
    ) -> None:
        self.embeddings: np.ndarray | None = None
        self.document_ids: list[str] = []
        if embeddings is not None and document_ids is not None:
            self.build(embeddings, document_ids)

    @property
    def is_ready(self) -> bool:
        return (
            self.embeddings is not None
            and len(self.document_ids) == len(self.embeddings)
            and len(self.document_ids) > 0
        )

    @staticmethod
    def _normalize(matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def build(
        self,
        embeddings: np.ndarray,
        document_ids: Sequence[str],
    ) -> None:
        matrix = self._normalize(embeddings)
        ids = [str(document_id) for document_id in document_ids]
        if len(matrix) != len(ids):
            raise ValueError("Embedding count must match document ID count.")
        if len(set(ids)) != len(ids):
            raise ValueError("Document IDs must be unique.")
        self.embeddings = matrix
        self.document_ids = ids

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.is_ready:
            raise RuntimeError("Embedding index is not ready.")

        query = self._normalize(np.asarray(query_embedding, dtype=np.float32))[0]
        scores = self.embeddings @ query
        k = min(max(1, int(top_k)), len(scores))
        indices = np.argpartition(-scores, kth=k - 1)[:k]
        indices = indices[np.argsort(-scores[indices])]
        return indices.astype(int), scores[indices].astype(float)

    def save(self, directory: str | Path, metadata: dict | None = None) -> None:
        if not self.is_ready:
            raise RuntimeError("Cannot save an empty index.")

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / self.EMBEDDINGS_FILE, self.embeddings)
        (directory / self.IDS_FILE).write_text(
            json.dumps(self.document_ids, indent=2),
            encoding="utf-8",
        )
        index_metadata = {
            "backend": "numpy",
            "document_count": len(self.document_ids),
            "embedding_dimension": int(self.embeddings.shape[1]),
            **(metadata or {}),
        }
        (directory / self.METADATA_FILE).write_text(
            json.dumps(index_metadata, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: str | Path) -> tuple["NumpyEmbeddingIndex", dict]:
        directory = Path(directory)
        embeddings_path = directory / cls.EMBEDDINGS_FILE
        ids_path = directory / cls.IDS_FILE
        metadata_path = directory / cls.METADATA_FILE

        if not all(path.exists() for path in [embeddings_path, ids_path, metadata_path]):
            raise FileNotFoundError("A complete saved index was not found.")

        embeddings = np.load(embeddings_path)
        document_ids = json.loads(ids_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return cls(embeddings, document_ids), metadata
