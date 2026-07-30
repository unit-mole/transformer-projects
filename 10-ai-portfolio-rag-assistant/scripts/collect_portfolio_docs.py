from __future__ import annotations

import argparse
from pathlib import Path
import shutil

ALLOWED_NAMES = {"README.md", "MODEL_CARD.md", "DATASET_CARD.md", "README_HUGGINGFACE.md", "README_GITHUB_PAGES.md", "README_VERCEL.md"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect public portfolio Markdown files into the safe raw corpus.")
    parser.add_argument("sources", nargs="+", type=Path, help="Public repository directories to scan")
    parser.add_argument("--output", type=Path, default=Path("data/raw_portfolio_docs"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    copied = 0
    for source in args.sources:
        for path in source.rglob("*.md"):
            if path.name not in ALLOWED_NAMES and not path.name.startswith("README"):
                continue
            relative = path.relative_to(source)
            target = args.output / source.name / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            copied += 1
    print(f"Collected {copied} public Markdown documents into {args.output}")

if __name__ == "__main__":
    main()
