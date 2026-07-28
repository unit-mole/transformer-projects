from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.advanced_evaluation import score_predictions
from src.benchmark_models import BenchmarkSpec, TransformerQABenchmarkRunner, evaluate_runner
from src.qasper_dataset import (
    build_controlled_context_variants,
    load_prepared_split,
    select_evaluation_sample,
)
from src.results_reporting import generate_complete_report, save_controlled_context_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark BERT and Longformer on QASPER.")
    parser.add_argument("--examples", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--longformer-max-length", type=int, default=2048)
    parser.add_argument("--longformer-stride", type=int, default=256)
    parser.add_argument("--skip-bert", action="store_true")
    parser.add_argument("--skip-base-longformer", action="store_true")
    parser.add_argument("--skip-finetuned", action="store_true")
    parser.add_argument("--controlled-base-examples", type=int, default=12)
    parser.add_argument("--skip-controlled", action="store_true")
    return parser.parse_args()


def _load_json(path: Path, fallback: dict) -> dict:
    if not path.exists():
        return fallback
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    validation_path = PROJECT_ROOT / "data" / "processed" / "qasper" / "qasper_validation_extractive.parquet"
    if not validation_path.exists():
        raise FileNotFoundError("Run scripts/prepare_qasper_dataset.py first.")
    validation = load_prepared_split(validation_path)
    sample = select_evaluation_sample(validation, args.examples, args.seed)
    sample.to_parquet(PROJECT_ROOT / "data" / "processed" / "qasper" / "qasper_evaluation_sample.parquet", index=False)

    specs: list[BenchmarkSpec] = []
    if not args.skip_bert:
        specs.append(
            BenchmarkSpec(
                name="BERT truncated 512",
                model_id="deepset/bert-base-cased-squad2",
                strategy="truncate",
                max_length=512,
                stride=0,
                inference_batch_size=4,
                description="Standard BERT QA baseline using only the first 512-token input.",
            )
        )
    if not args.skip_base_longformer:
        specs.append(
            BenchmarkSpec(
                name="Longformer SQuAD sliding windows",
                model_id="valhalla/longformer-base-4096-finetuned-squadv1",
                strategy="sliding",
                max_length=args.longformer_max_length,
                stride=args.longformer_stride,
                inference_batch_size=1,
                description="Published Longformer QA checkpoint evaluated across the full document.",
            )
        )
    fine_tuned_path = PROJECT_ROOT / "models" / "qasper-longformer"
    if not args.skip_finetuned and (fine_tuned_path / "config.json").exists():
        specs.append(
            BenchmarkSpec(
                name="Longformer QASPER fine-tuned",
                model_id=str(fine_tuned_path),
                strategy="sliding",
                max_length=args.longformer_max_length,
                stride=args.longformer_stride,
                inference_batch_size=1,
                description="Project-trained Longformer checkpoint fine-tuned on extractive QASPER examples.",
            )
        )

    scored_by_model: dict[str, pd.DataFrame] = {}
    for spec in specs:
        print(f"\nEvaluating {spec.name} ({spec.model_id})")
        runner = TransformerQABenchmarkRunner(spec)
        predictions = evaluate_runner(
            runner,
            sample,
            progress_callback=lambda done, total, name: print(
                f"\r{name}: {done}/{total}", end="", flush=True
            ),
        )
        print()
        scored_by_model[spec.name] = score_predictions(predictions)
        runner.unload()

    dataset_summary = _load_json(
        PROJECT_ROOT / "outputs" / "qasper_dataset_summary.json",
        {"status": "unknown", "dataset": "QASPER v0.3"},
    )
    training_summary = _load_json(
        PROJECT_ROOT / "outputs" / "training_summary.json",
        {"status": "not_run"},
    )
    manifest = generate_complete_report(
        PROJECT_ROOT,
        scored_by_model,
        dataset_summary,
        training_summary,
    )

    if not args.skip_controlled and specs:
        from transformers import AutoTokenizer

        control_tokenizer = AutoTokenizer.from_pretrained(
            "valhalla/longformer-base-4096-finetuned-squadv1", use_fast=True
        )
        controlled = build_controlled_context_variants(
            sample,
            control_tokenizer,
            maximum_base_examples=args.controlled_base_examples,
            seed=args.seed,
        )
        controlled.to_parquet(
            PROJECT_ROOT / "data" / "processed" / "qasper" / "qasper_controlled_context_sample.parquet",
            index=False,
        )
        controlled_scored: dict[str, pd.DataFrame] = {}
        for spec in specs:
            print(f"\nControlled context evaluation: {spec.name}")
            runner = TransformerQABenchmarkRunner(spec)
            predictions = evaluate_runner(runner, controlled)
            controlled_scored[spec.name] = score_predictions(predictions)
            runner.unload()
        controlled_comparison = save_controlled_context_results(
            controlled_scored, PROJECT_ROOT / "outputs"
        )
        manifest["controlled_context_rows"] = int(len(controlled_comparison))
        (PROJECT_ROOT / "outputs" / "evaluation_manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str), encoding="utf-8"
        )

    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()
