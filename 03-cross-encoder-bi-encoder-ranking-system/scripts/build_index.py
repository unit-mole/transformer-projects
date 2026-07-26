from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ranking_engine import TwoStageRankingEngine
from src.settings import Settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the lightweight NumPy index.")
    parser.add_argument("--force", action="store_true", help="Rebuild an existing index.")
    args = parser.parse_args()

    engine = TwoStageRankingEngine.from_settings(Settings.from_yaml())
    elapsed_ms = engine.prepare_index(force_rebuild=args.force, save=True)
    print(
        f"Index ready: {len(engine.index.document_ids)} documents, "
        f"{elapsed_ms:.2f} ms, path={engine.settings.index_dir}"
    )


if __name__ == "__main__":
    main()
