"""Promote a reviewed experiment into the repository's deployment locations."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def promote_experiment(
    *,
    project_root: str | Path,
    experiment_dir: str | Path,
    human_review_completed: bool,
) -> Dict[str, Any]:
    """Copy trained/reviewed artifacts into standard deployment paths.

    The explicit human-review flag prevents an unreviewed synthetic dataset or
    evaluation from being presented as a finished portfolio result.
    """
    if not human_review_completed:
        raise ValueError("Set human_review_completed=True only after reviewing dataset and generated responses.")

    root = Path(project_root)
    experiment = Path(experiment_dir)
    training = experiment / "training"
    evaluation = experiment / "evaluation"
    adapter_source = training / "lora_adapter"
    tokenizer_source = training / "tokenizer"
    if not (adapter_source / "adapter_config.json").exists():
        raise FileNotFoundError(f"No trained adapter found at {adapter_source}")
    if not (evaluation / "comparison" / "base_vs_lora_comparison.json").exists():
        raise FileNotFoundError("Base-versus-LoRA evaluation has not been completed.")

    _copy_tree(adapter_source, root / "models" / "lora_adapter")
    if tokenizer_source.exists():
        _copy_tree(tokenizer_source, root / "models" / "tokenizer")

    selected_files = {
        training / "model_metadata.json": root / "models" / "model_metadata.json",
        training / "training_curve.png": root / "outputs" / "training_curve.png",
        training / "training_log_history.json": root / "outputs" / "training_log_history.json",
        evaluation / "base_model" / "metrics.json": root / "outputs" / "base_model_metrics.json",
        evaluation / "lora_model" / "metrics.json": root / "outputs" / "lora_model_metrics.json",
        evaluation / "comparison" / "base_vs_lora_comparison.json": root / "outputs" / "base_vs_lora_comparison.json",
        evaluation / "comparison" / "per_example_base_vs_lora.csv": root / "outputs" / "per_example_base_vs_lora.csv",
        evaluation / "comparison" / "before_after_finetuning_examples.md": root / "outputs" / "before_after_finetuning_examples.md",
        evaluation / "comparison" / "base_vs_lora_metric_comparison.png": root / "outputs" / "base_vs_lora_metric_comparison.png",
        evaluation / "evaluation_manifest.json": root / "outputs" / "evaluation_manifest.json",
    }
    copied = []
    for source, destination in selected_files.items():
        if source.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied.append(str(destination.relative_to(root)))

    manifest_files = []
    for path in sorted((root / "models" / "lora_adapter").rglob("*")):
        if path.is_file():
            manifest_files.append({
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    for relative in copied:
        path = root / relative
        manifest_files.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})

    manifest = {
        "status": "portfolio_release_candidate",
        "human_review_completed": True,
        "experiment_dir": str(experiment),
        "adapter_mode": "local_lora_adapter",
        "files": manifest_files,
    }
    (root / "outputs" / "release_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
