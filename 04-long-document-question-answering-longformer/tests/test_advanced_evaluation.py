from __future__ import annotations

import json

import pandas as pd

from src.advanced_evaluation import (
    evidence_recovered,
    evidence_token_recall,
    exact_match,
    normalize_answer,
    score_predictions,
    token_f1,
)


def test_squad_style_normalization() -> None:
    assert normalize_answer("The Supplier's Process.") == "suppliers process"
    assert exact_match("The CAPA owner", "CAPA owner") == 1.0


def test_token_f1_partial_credit() -> None:
    score = token_f1("supplier ultraviolet curing time", "variation in supplier ultraviolet curing time")
    assert 0.70 < score < 1.0


def test_evidence_metrics() -> None:
    references = ["The confirmed root cause was variation in ultraviolet curing time."]
    predicted = "Root cause: variation in ultraviolet curing time. Corrective action followed."
    assert evidence_token_recall(predicted, references) > 0.5
    assert evidence_recovered(predicted, references) == 1.0


def test_score_predictions_handles_multiple_references() -> None:
    frame = pd.DataFrame(
        [
            {
                "example_id": "x",
                "reference_answers_json": json.dumps(["June 18, 2026", "18 June 2026"]),
                "reference_evidence_json": json.dumps(["Approved on June 18, 2026."]),
                "predicted_answer": "June 18, 2026",
                "predicted_evidence": "The review was approved on June 18, 2026.",
                "document_token_count": 1800,
                "answer_token_position": 1400,
                "confidence_proxy": 0.2,
                "window_count": 2,
                "latency_seconds": 0.4,
                "error": "",
            }
        ]
    )
    scored = score_predictions(frame)
    assert scored.loc[0, "exact_match"] == 1.0
    assert scored.loc[0, "context_length_bucket"] == "1025-2048"
    assert bool(scored.loc[0, "answer_beyond_512"]) is True
