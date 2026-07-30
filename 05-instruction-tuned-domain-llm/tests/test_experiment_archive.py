from __future__ import annotations

import csv
import json
from pathlib import Path

from src.experiment_archive import archive_experiment


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_archive_is_non_destructive_and_creates_manifest(tmp_path: Path) -> None:
    experiment = tmp_path / "outputs" / "experiments" / "run1"
    (experiment / "training").mkdir(parents=True)
    (experiment / "training" / "model_metadata.json").write_text(
        json.dumps({"status": "completed", "base_model": "google/flan-t5-base"}),
        encoding="utf-8",
    )
    _write_csv(
        experiment / "evaluation" / "comparison" / "per_example_base_vs_lora.csv",
        [{"id": "1", "human_preferred_model": "lora"}],
    )
    _write_csv(
        experiment / "evaluation" / "base_model" / "manual_review_results.csv",
        [{"id": "1", "human_factuality_1_to_5": 2}],
    )
    _write_csv(
        experiment / "evaluation" / "lora_model" / "manual_review_results.csv",
        [{"id": "1", "human_factuality_1_to_5": 3}],
    )

    result = archive_experiment(
        experiment,
        tmp_path / "archives",
        archive_label="experiment_1",
        create_full_zip=True,
    )
    assert result["status"] == "archived_without_deleting_source"
    assert experiment.exists()
    archive = Path(result["archive_directory"])
    assert (archive / "experiment_manifest.json").exists()
    assert (archive / "EXPERIMENT_1_CARD.md").exists()
    assert Path(result["full_zip"]).exists()
