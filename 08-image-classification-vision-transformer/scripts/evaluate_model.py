from __future__ import annotations
import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained image classifier.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    raise SystemExit("Connect the checkpoint to the provided dataset loader, collect y_true/y_pred, and call src.model_evaluation.save_evaluation. No metrics are fabricated by this scaffold.")

if __name__ == "__main__":
    main()
