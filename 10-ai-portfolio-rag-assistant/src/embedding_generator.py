from __future__ import annotations

from collections import Counter
import math
import re
from typing import Iterable

import numpy as np

TOKEN_PATTERN = re.compile(r"[a-z0-9+#.\-]+")
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "i", "in", "is",
    "it", "of", "on", "or", "that", "the", "this", "to", "was", "were", "what", "which", "who",
    "with", "my", "show", "does",
}


def fnv1a(token: str) -> int:
    value = 0x811C9DC5
    for char in token:
        value ^= ord(char)
        value = (value * 0x01000193) & 0xFFFFFFFF
    return value


def local_hash_embedding(text: str, dimension: int = 384) -> np.ndarray:
    tokens = [
        token
        for token in TOKEN_PATTERN.findall(text.lower())
        if len(token) > 1 and token not in STOP_WORDS
    ]
    vector = np.zeros(dimension, dtype=np.float32)
    for token, count in Counter(tokens).items():
        hashed = fnv1a(token)
        vector[hashed % dimension] += (
            (1.0 if hashed & 1 == 0 else -1.0) * (1.0 + math.log(count))
        )
    norm = np.linalg.norm(vector)
    return vector if norm == 0 else vector / norm


def sentence_transformer_embeddings(
    texts: Iterable[str],
    model_name: str,
    device: str | None = None,
    batch_size: int = 64,
    prefix: str = "",
) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Install sentence-transformers before generating Transformer embeddings.") from exc

    model = SentenceTransformer(model_name, device=device)
    prepared = [f"{prefix}{text}" for text in texts]
    return model.encode(
        prepared,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=batch_size,
        convert_to_numpy=True,
    ).astype(np.float32)


def minilm_embeddings(
    texts: Iterable[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str | None = None,
    batch_size: int = 64,
) -> np.ndarray:
    return sentence_transformer_embeddings(
        texts,
        model_name=model_name,
        device=device,
        batch_size=batch_size,
    )
