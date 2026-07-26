from __future__ import annotations
import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a matched ViT-vs-CNN/ResNet comparison.")
    parser.add_argument("--vit-checkpoint", required=True)
    parser.add_argument("--cnn-checkpoint", required=True)
    parser.add_argument("--output", default="outputs/vit_vs_cnn_comparison.csv")
    args = parser.parse_args()
    for value in [args.vit_checkpoint, args.cnn_checkpoint]:
        if not Path(value).exists():
            raise FileNotFoundError(value)
    raise SystemExit("Evaluate both checkpoints on the same split and benchmark with identical warm-up/runs before writing the comparison CSV.")

if __name__ == "__main__":
    main()
