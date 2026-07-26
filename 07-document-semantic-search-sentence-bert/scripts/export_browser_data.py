#!/usr/bin/env python
"""Copy validated processed artifacts into web/data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.export_for_browser import export_browser_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=PROJECT_ROOT / "data/processed")
    parser.add_argument("--web-data-dir", type=Path, default=PROJECT_ROOT / "web/data")
    args = parser.parse_args()
    copied = export_browser_data(args.processed_dir, args.web_data_dir)
    print("Exported browser artifacts:")
    for path in copied:
        print(f"- {path}")


if __name__ == "__main__":
    main()
