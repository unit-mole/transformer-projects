from __future__ import annotations

from typing import Sequence

import numpy as np


class CrossEncoderModel:
    """Lazy MS MARCO CrossEncoder wrapper for second-stage reranking."""

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        model: object | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model = model

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for reranking. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc

        self._model = CrossEncoder(self.model_name, device=self.device)

    def score(self, query: str, documents: Sequence[str]) -> np.ndarray:
        self.load()
        if not documents:
            return np.array([], dtype=np.float32)

        pairs = [(query, document) for document in documents]
        scores = self._model.predict(
            pairs,
            batch_size=min(32, max(1, len(pairs))),
            show_progress_bar=False,
        )
        return np.asarray(scores, dtype=np.float32).reshape(-1)
