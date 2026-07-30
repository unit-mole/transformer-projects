"""Archive a completed experiment without changing or deleting the source run."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _numeric_mean(rows: Iterable[dict[str, str]], field: str) -> float | None:
    values: list[float] = []
    for row in rows:
        value = str(row.get(field, "")).strip()
        if not value:
            continue
        try:
            values.append(float(value))
        except ValueError:
            continue
    return round(sum(values) / len(values), 4) if values else None


def summarize_experiment(experiment_dir: str | Path) -> Dict[str, Any]:
    experiment = Path(experiment_dir)
    training = _read_json(experiment / "training" / "model_metadata.json")
    evaluation = _read_json(experiment / "evaluation" / "evaluation_manifest.json")
    comparison_rows = _read_csv(
        experiment / "evaluation" / "comparison" / "per_example_base_vs_lora.csv"
    )
    base_review = _read_csv(
        experiment / "evaluation" / "base_model" / "manual_review_results.csv"
    )
    lora_review = _read_csv(
        experiment / "evaluation" / "lora_model" / "manual_review_results.csv"
    )

    preferences = {"base": 0, "lora": 0, "tie": 0, "missing": 0}
    for row in comparison_rows:
        choice = str(row.get("human_preferred_model", "")).strip().lower()
        if choice in preferences:
            preferences[choice] += 1
        else:
            preferences["missing"] += 1

    return {
        "experiment_name": experiment.name,
        "source_path": str(experiment.resolve()),
        "training_status": training.get("status"),
        "base_model": training.get("base_model"),
        "fine_tuning_method": training.get("fine_tuning_method"),
        "dataset_path": training.get("dataset_path"),
        "dataset_split_sizes": training.get("dataset_split_sizes", {}),
        "validation_loss": (training.get("validation_metrics") or {}).get("validation_loss"),
        "test_loss": (training.get("test_metrics") or {}).get("test_loss"),
        "validation_perplexity": training.get("validation_perplexity"),
        "best_checkpoint": training.get("best_checkpoint"),
        "trainable_parameters": training.get("trainable_parameters"),
        "trainable_percentage": training.get("trainable_percentage"),
        "evaluation_status": evaluation.get("status"),
        "benchmark_examples": (evaluation.get("comparison") or {}).get("benchmark_examples"),
        "human_preference_counts": preferences,
        "base_human_review": {
            "rows": len(base_review),
            "mean_factuality": _numeric_mean(base_review, "human_factuality_1_to_5"),
            "mean_relevance": _numeric_mean(base_review, "human_relevance_1_to_5"),
            "mean_clarity": _numeric_mean(base_review, "human_clarity_1_to_5"),
            "mean_instruction_following": _numeric_mean(
                base_review, "human_instruction_following_1_to_5"
            ),
        },
        "lora_human_review": {
            "rows": len(lora_review),
            "mean_factuality": _numeric_mean(lora_review, "human_factuality_1_to_5"),
            "mean_relevance": _numeric_mean(lora_review, "human_relevance_1_to_5"),
            "mean_clarity": _numeric_mean(lora_review, "human_clarity_1_to_5"),
            "mean_instruction_following": _numeric_mean(
                lora_review, "human_instruction_following_1_to_5"
            ),
        },
        "decision": "preserved_as_experiment_1_not_promoted",
        "reason": (
            "Training completed and LoRA improved some relative metrics, but human review "
            "found weak factuality, circular answers, and hallucination risk."
        ),
    }


def _experiment_card(summary: Dict[str, Any]) -> str:
    pref = summary.get("human_preference_counts", {})
    lora_review = summary.get("lora_human_review", {})
    return f"""# Experiment 1 — Initial FLAN-T5-base LoRA Run

## Status

**Preserved, reviewed, and intentionally not promoted.**

## Purpose

This run established the first complete training and evaluation baseline for the
ML/Data Science Learning Assistant. It is retained as evidence of an honest
iteration cycle rather than deleted after weak response-quality findings.

## Core results

- Base model: `{summary.get('base_model')}`
- Fine-tuning: `{summary.get('fine_tuning_method')}`
- Validation loss: `{summary.get('validation_loss')}`
- Test loss: `{summary.get('test_loss')}`
- Validation perplexity: `{summary.get('validation_perplexity')}`
- Trainable percentage: `{summary.get('trainable_percentage')}`
- LoRA preferred by human review: `{pref.get('lora', 0)}`
- Base preferred by human review: `{pref.get('base', 0)}`
- Ties: `{pref.get('tie', 0)}`
- LoRA mean factuality: `{lora_review.get('mean_factuality')}` / 5
- LoRA mean relevance: `{lora_review.get('mean_relevance')}` / 5
- LoRA mean clarity: `{lora_review.get('mean_clarity')}` / 5
- LoRA mean instruction following: `{lora_review.get('mean_instruction_following')}` / 5

## Decision

The adapter was not promoted because human evaluation found technically weak,
circular, incomplete, or hallucinated answers. Experiment 2 therefore changes
the supervision quality and keeps the held-out benchmark unchanged.

## Recruiter-facing value

This experiment demonstrates reproducible GPU training, LoRA/PEFT, held-out
benchmarking, automated metrics, human review, release gating, and evidence-based
iteration rather than selective reporting.
"""


def archive_experiment(
    experiment_dir: str | Path,
    archive_root: str | Path,
    *,
    archive_label: str = "experiment_1_initial_lora",
    create_full_zip: bool = True,
    extra_files: Iterable[str | Path] | None = None,
) -> Dict[str, Any]:
    """Create a non-destructive archive, checksums, summary, and optional full ZIP."""
    source = Path(experiment_dir).resolve()
    if not source.exists():
        raise FileNotFoundError(f"Experiment folder was not found: {source}")

    archive_base = Path(archive_root).resolve() / archive_label
    archive_base.mkdir(parents=True, exist_ok=True)

    summary = summarize_experiment(source)
    summary["archived_at_utc"] = datetime.now(timezone.utc).isoformat()
    summary["archive_path"] = str(archive_base)
    (archive_base / "experiment_1_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    (archive_base / "EXPERIMENT_1_CARD.md").write_text(
        _experiment_card(summary), encoding="utf-8"
    )

    manifest: list[dict[str, Any]] = []
    for path in sorted(source.rglob("*")):
        if path.is_file():
            manifest.append(
                {
                    "path": str(path.relative_to(source)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )

    extra_manifest: list[dict[str, Any]] = []
    for extra in extra_files or []:
        path = Path(extra).resolve()
        if path.exists() and path.is_file():
            destination = archive_base / "notebook_snapshots" / path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            extra_manifest.append(
                {
                    "path": str(destination.relative_to(archive_base)),
                    "bytes": destination.stat().st_size,
                    "sha256": sha256_file(destination),
                }
            )

    manifest_payload = {
        "source_experiment": str(source),
        "source_still_exists": source.exists(),
        "file_count": len(manifest),
        "files": manifest,
        "extra_files": extra_manifest,
    }
    (archive_base / "experiment_manifest.json").write_text(
        json.dumps(manifest_payload, indent=2), encoding="utf-8"
    )
    checksum_lines = [f"{item['sha256']}  {item['path']}" for item in manifest]
    (archive_base / "sha256sums.txt").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )

    zip_path: Path | None = None
    if create_full_zip:
        zip_path = archive_base / f"{archive_label}_full.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(source.rglob("*")):
                if path.is_file():
                    zf.write(path, arcname=f"{source.name}/{path.relative_to(source)}")
            for extra in extra_files or []:
                path = Path(extra).resolve()
                if path.exists() and path.is_file():
                    zf.write(path, arcname=f"notebook_snapshots/{path.name}")

    result = {
        "status": "archived_without_deleting_source",
        "source_experiment": str(source),
        "source_still_exists": source.exists(),
        "archive_directory": str(archive_base),
        "full_zip": str(zip_path) if zip_path else None,
        "summary_file": str(archive_base / "experiment_1_summary.json"),
        "card_file": str(archive_base / "EXPERIMENT_1_CARD.md"),
        "manifest_file": str(archive_base / "experiment_manifest.json"),
        "checksums_file": str(archive_base / "sha256sums.txt"),
        "file_count": len(manifest),
    }
    (archive_base / "archive_result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result
