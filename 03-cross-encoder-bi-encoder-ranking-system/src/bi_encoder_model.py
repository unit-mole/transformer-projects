from __future__ import annotations

from typing import Sequence

import numpy as np


class BiEncoderModel:
    """Lazy SentenceTransformer wrapper used for query and document embeddings."""

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
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for Transformer inference. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc

        self._model = SentenceTransformer(self.model_name, device=self.device)

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        self.load()
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        embeddings = self._model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        return self.encode([query], batch_size=1, normalize_embeddings=True)[0]
