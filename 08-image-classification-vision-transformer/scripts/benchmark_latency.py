from __future__ import annotations
import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a real checkpoint with warm-up and repeated runs.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=30)
    args = parser.parse_args()
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(args.checkpoint)
    raise SystemExit("Load the checkpoint and input tensor, then pass a zero-argument inference callable to src.latency_benchmark.benchmark.")

if __name__ == "__main__":
    main()
