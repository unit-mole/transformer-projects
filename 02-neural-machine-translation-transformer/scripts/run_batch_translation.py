from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.batch_translation import translate_csv  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Translate a CSV text column.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--text-column", required=True)
    parser.add_argument(
        "--direction",
        default="auto",
        choices=["auto", "en_hi", "hi_en"],
    )
    parser.add_argument("--max-rows", type=int, default=100)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    _, output_path, summary = translate_csv(
        args.input,
        text_column=args.text_column,
        direction=args.direction,
        max_rows=args.max_rows,
        output_path=args.output,
    )
    print(json.dumps({"output": output_path, **summary}, indent=2))
