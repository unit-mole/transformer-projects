from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vqa.dataset_loader import load_vqa_csv
from vqa.evaluation import evaluate_records
from vqa.inference_pipeline import VQAInferencePipeline

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data/sample_vqa_pairs.csv"))
    parser.add_argument("--output", default=str(ROOT / "outputs/model_metrics.json"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    records = load_vqa_csv(args.data)
    if args.limit:
        records = records[: args.limit]
    pipeline = VQAInferencePipeline(device=args.device)
    predictions = []
    for row in records:
        result = pipeline.predict(ROOT / row["image_path"], row["question"])
        predictions.append({**row, "prediction": result.answer, "latency_seconds": result.latency_seconds})

    metrics = evaluate_records(predictions)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    prediction_csv = output.with_name("prediction_examples.csv")
    if predictions:
        with prediction_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=predictions[0].keys())
            writer.writeheader()
            writer.writerows(predictions)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
