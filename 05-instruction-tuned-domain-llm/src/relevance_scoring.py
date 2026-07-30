"""Transparent heuristic response-relevance scoring."""
from __future__ import annotations

import re
from typing import Dict


def _tokens(text: str) -> set[str]:
    stop = {"the", "a", "an", "and", "or", "to", "of", "in", "is", "are", "for", "with", "what", "how", "explain"}
    return {t for t in re.findall(r"[a-z0-9_+-]+", text.lower()) if len(t) > 2 and t not in stop}


def lexical_relevance(prompt: str, response: str) -> float:
    prompt_tokens = _tokens(prompt)
    response_tokens = _tokens(response)
    if not prompt_tokens or not response_tokens:
        return 0.0
    return round(len(prompt_tokens & response_tokens) / len(prompt_tokens), 4)


def embedding_relevance(prompt: str, response: str) -> float:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError as exc:
        raise ImportError("Install scikit-learn for TF-IDF relevance scoring.") from exc
    matrix = TfidfVectorizer(ngram_range=(1, 2), stop_words="english").fit_transform([prompt, response])
    return round(float(cosine_similarity(matrix[0], matrix[1])[0, 0]), 4)


def score_relevance(prompt: str, response: str) -> Dict[str, float | str]:
    lexical = lexical_relevance(prompt, response)
    cosine = embedding_relevance(prompt, response)
    combined = round(0.4 * lexical + 0.6 * cosine, 4)
    return {
        "lexical_relevance": lexical,
        "tfidf_cosine_relevance": cosine,
        "combined_relevance": combined,
        "interpretation": "heuristic_only_not_absolute_truth",
    }
