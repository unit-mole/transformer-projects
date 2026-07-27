from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .clip_model import ClipEncoder
from .retrieval_engine import RetrievalResult, rank_embeddings
from .text_preprocessing import clean_text


@dataclass
class ClipRetrievalPipeline:
    encoder: ClipEncoder
    image_ids: Sequence[str]
    gallery_embeddings: np.ndarray

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        cleaned = clean_text(query, field_name="query", max_length=240)
        query_embedding = self.encoder.encode_text([cleaned])[0]
        return rank_embeddings(query_embedding, self.gallery_embeddings, self.image_ids, top_k)

    @classmethod
    def from_files(cls, image_ids: Sequence[str], embedding_path: str | Path, encoder: ClipEncoder | None = None) -> "ClipRetrievalPipeline":
        embeddings = np.load(embedding_path)
        return cls(encoder or ClipEncoder(), image_ids, embeddings)
