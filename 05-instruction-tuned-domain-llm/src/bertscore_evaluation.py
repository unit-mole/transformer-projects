"""BERTScore evaluation where reference answers are available."""
from __future__ import annotations

from typing import Dict, Sequence


def calculate_bertscore(
    predictions: Sequence[str],
    references: Sequence[str],
    *,
    lang: str = "en",
    model_type: str | None = None,
) -> Dict[str, object]:
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length.")
    if not predictions:
        return {"precision": [], "recall": [], "f1": [], "average_f1": None, "status": "no_examples"}
    try:
        from bert_score import score
    except ImportError as exc:
        raise ImportError("Install bert-score to calculate BERTScore.") from exc

    kwargs = {"lang": lang, "verbose": False}
    if model_type:
        kwargs["model_type"] = model_type
    precision, recall, f1 = score(list(predictions), list(references), **kwargs)
    p = [round(float(x), 6) for x in precision]
    r = [round(float(x), 6) for x in recall]
    f = [round(float(x), 6) for x in f1]
    return {
        "precision": p,
        "recall": r,
        "f1": f,
        "average_precision": round(sum(p) / len(p), 6),
        "average_recall": round(sum(r) / len(r), 6),
        "average_f1": round(sum(f) / len(f), 6),
        "status": "completed",
        "limitation": "semantic_similarity_not_factual_correctness",
    }
