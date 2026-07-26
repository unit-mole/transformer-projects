"""Classification metrics and artifact writers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def evaluate_predictions(y_true: Iterable[int], y_pred: Iterable[int], class_names: list[str]) -> dict:
    truth = np.asarray(list(y_true))
    pred = np.asarray(list(y_pred))
    if truth.size == 0 or truth.shape != pred.shape:
        raise ValueError("y_true and y_pred must be non-empty and have matching shapes.")
    report = classification_report(truth, pred, target_names=class_names, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(truth, pred)),
        "macro_f1": float(f1_score(truth, pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(truth, pred).tolist(),
        "classification_report": report,
    }


def save_evaluation(metrics: dict, output_dir: str | Path, class_names: list[str]) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "model_metrics.json").write_text(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2), encoding="utf-8")
    rows = []
    for name in class_names:
        values = metrics["classification_report"][name]
        rows.append({"class": name, "precision": values["precision"], "recall": values["recall"], "f1_score": values["f1-score"], "support": values["support"]})
    pd.DataFrame(rows).to_csv(output / "classification_report.csv", index=False)
