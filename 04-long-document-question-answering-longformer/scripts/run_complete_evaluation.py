from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    print("\n$", " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Project 04 dataset, training and evaluation workflow.")
    parser.add_argument("--profile", choices=["smoke", "portfolio", "full", "high-vram"], default="portfolio")
    parser.add_argument("--examples", type=int, default=120)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    python = sys.executable
    prepare = [python, "scripts/prepare_qasper_dataset.py"]
    if args.force_download:
        prepare.append("--force-download")
    run(prepare)
    if not args.skip_training:
        run([python, "scripts/fine_tune_longformer_qasper.py", "--profile", args.profile])
    run([python, "scripts/evaluate_qasper_benchmarks.py", "--examples", str(args.examples)])


if __name__ == "__main__":
    main()
