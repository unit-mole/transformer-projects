"""Optional BERTScore evaluation for examples that have reference answers."""

from __future__ import annotations

from typing import Sequence


def calculate_bertscore(predictions: Sequence[str], references: Sequence[str], lang: str = "en") -> dict:
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    if not predictions:
        return {"precision": [], "recall": [], "f1": [], "average_f1": None}
    try:
        from bert_score import score
    except ImportError as exc:
        raise RuntimeError("Install bert-score to calculate BERTScore.") from exc
    precision, recall, f1 = score(list(predictions), list(references), lang=lang, verbose=False)
    return {
        "precision": precision.cpu().tolist(),
        "recall": recall.cpu().tolist(),
        "f1": f1.cpu().tolist(),
        "average_precision": float(precision.mean()),
        "average_recall": float(recall.mean()),
        "average_f1": float(f1.mean()),
        "note": "BERTScore measures semantic similarity, not factual correctness.",
    }
