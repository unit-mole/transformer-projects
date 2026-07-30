from __future__ import annotations

import json
from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_ROOT / "data/processed"
TARGET = PROJECT_ROOT / "public/data"
REQUIRED = ["document_chunks.json", "embeddings.json", "metadata.json", "evaluation_questions.json"]


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED:
        source = SOURCE / name
        if not source.exists():
            raise FileNotFoundError(f"Missing required artifact: {source}")
        json.loads(source.read_text(encoding="utf-8"))
        shutil.copy2(source, TARGET / name)
    print(f"Exported {len(REQUIRED)} Vercel-ready files to {TARGET}")

if __name__ == "__main__":
    main()
