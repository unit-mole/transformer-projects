from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

def main() -> None:
    csv_path = DATA / "sample_vqa_pairs.csv"
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    missing = [row["image_path"] for row in rows if not (ROOT / row["image_path"]).exists()]
    if missing:
        raise FileNotFoundError(f"Missing sample images: {missing}")
    print(f"Validated {len(rows)} sample VQA records.")

if __name__ == "__main__":
    main()
