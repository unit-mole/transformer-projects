from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.latency_benchmark import benchmark_latency
from src.ranking_engine import TwoStageRankingEngine
from src.settings import Settings
from src.visualization import plot_latency


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark search latency.")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    engine = TwoStageRankingEngine.from_settings(Settings.from_yaml())
    queries = engine.sample_queries[:5]
    frame = benchmark_latency(engine, queries=queries, repeats=args.repeats)

    outputs = PROJECT_ROOT / "outputs"
    outputs.mkdir(exist_ok=True)
    frame.to_csv(outputs / "latency_results.csv", index=False)
    (outputs / "latency_results.json").write_text(
        frame.to_json(orient="records", indent=2),
        encoding="utf-8",
    )
    plot_latency(frame, outputs / "latency_by_top_k.png")
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
