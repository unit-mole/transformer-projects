from __future__ import annotations
import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a compact DeiT/ViT on CIFAR-10.")
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10"])
    parser.add_argument("--model", default="facebook/deit-tiny-patch16-224")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--output", default="models/vit_model")
    args = parser.parse_args()
    raise SystemExit(
        "Training requires experiment-specific compute choices. The reusable loaders and model modules are ready. "
        f"Configure batch size/optimizer, then save the real checkpoint to {Path(args.output)}. "
        "Do not run training inside GitHub Actions."
    )

if __name__ == "__main__":
    main()
