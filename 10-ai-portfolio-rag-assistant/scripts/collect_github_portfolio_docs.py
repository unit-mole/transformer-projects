from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.github_collector import collect_repositories, load_repository_configs


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect public portfolio Markdown files from GitHub.")
    parser.add_argument("--config", type=Path, default=ROOT / "config/portfolio_repositories.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/raw_portfolio_docs")
    parser.add_argument("--clean", action="store_true", help="Delete the existing output directory first.")
    args = parser.parse_args()

    if args.clean and args.output.exists():
        import shutil
        shutil.rmtree(args.output)

    configs = load_repository_configs(args.config)
    token = os.getenv("GITHUB_TOKEN") or None
    written = collect_repositories(configs, args.output, token=token)
    print(f"Collected {len(written)} public Markdown files into {args.output}")
    if not token:
        print("GITHUB_TOKEN was not set. Public GitHub rate limits may apply.")


if __name__ == "__main__":
    main()
