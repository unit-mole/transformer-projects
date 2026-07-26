"""Documented heuristic response-relevance scoring."""

from __future__ import annotations

from typing import Any


def score_relevance(instruction: str, response: str, reference: str = "") -> dict[str, Any]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    texts = [instruction, response] if not reference else [instruction, reference, response]
    matrix = TfidfVectorizer(ngram_range=(1, 2), stop_words="english").fit_transform(texts)
    prompt_similarity = float(cosine_similarity(matrix[0], matrix[-1])[0, 0])
    reference_similarity = None
    if reference:
        reference_similarity = float(cosine_similarity(matrix[1], matrix[-1])[0, 0])
    combined = prompt_similarity if reference_similarity is None else (0.35 * prompt_similarity + 0.65 * reference_similarity)
    return {
        "relevance_score": round(combined, 4),
        "prompt_similarity": round(prompt_similarity, 4),
        "reference_similarity": None if reference_similarity is None else round(reference_similarity, 4),
        "note": "TF-IDF similarity is a heuristic, not an absolute quality judgment.",
    }
