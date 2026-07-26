from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .text_preprocessing import clean_text, split_sentences


def lead3_summary(text: str, max_sentences: int = 3) -> str:
    sentences = split_sentences(text)
    return clean_text(" ".join(sentences[:max_sentences]))


def textrank_summary(text: str, max_sentences: int = 3) -> str:
    """A lightweight TextRank-style sentence-centrality baseline."""
    sentences = split_sentences(text)
    if len(sentences) <= max_sentences:
        return clean_text(" ".join(sentences))
    try:
        matrix = TfidfVectorizer(stop_words="english").fit_transform(sentences)
    except ValueError:
        return lead3_summary(text, max_sentences=max_sentences)
    similarity = cosine_similarity(matrix)
    np.fill_diagonal(similarity, 0.0)
    scores = similarity.sum(axis=1)
    selected = sorted(np.argsort(scores)[-max_sentences:].tolist())
    return clean_text(" ".join(sentences[index] for index in selected))
