from __future__ import annotations

import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data"
WEB_DATA = PROJECT_ROOT / "web" / "public" / "data"


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    documents = read_csv(DATA / "sample_documents.csv")
    queries = read_csv(DATA / "sample_queries.csv")
    qrels = read_csv(DATA / "sample_qrels.csv")

    for row in qrels:
        row["relevance"] = int(row["relevance"])

    write_json(WEB_DATA / "sample_documents.json", documents)
    write_json(WEB_DATA / "sample_queries.json", queries)
    write_json(WEB_DATA / "sample_qrels.json", qrels)

    print(f"Synced {len(documents)} documents.")
    print(f"Synced {len(queries)} queries.")
    print(f"Synced {len(qrels)} qrels.")


if __name__ == "__main__":
    main()
