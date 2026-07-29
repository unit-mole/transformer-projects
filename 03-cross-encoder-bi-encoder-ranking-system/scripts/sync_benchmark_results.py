from __future__ import annotations

import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LATEST = PROJECT_ROOT / "outputs" / "benchmark" / "latest"
MODEL_HUB = PROJECT_ROOT / "model_hub" / "pipeline-card"


def main() -> None:
    summary_path = LATEST / "benchmark_summary.json"
    bootstrap_path = LATEST / "bootstrap_significance.json"
    metadata_path = LATEST / "dataset_metadata.json"
    portfolio_path = LATEST / "PORTFOLIO_RESULTS.md"
    required = [summary_path, bootstrap_path, metadata_path, portfolio_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(
            "Run the benchmark first. Missing files:\n" + "\n".join(missing)
        )

    payload = {
        "status": "completed",
        "summary": json.loads(summary_path.read_text(encoding="utf-8")),
        "bootstrap": json.loads(bootstrap_path.read_text(encoding="utf-8")),
        "datasets": json.loads(metadata_path.read_text(encoding="utf-8")),
        "source": "outputs/benchmark/latest",
    }
    MODEL_HUB.mkdir(parents=True, exist_ok=True)
    (MODEL_HUB / "evaluation_results.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    shutil.copy2(portfolio_path, PROJECT_ROOT / "BENCHMARK_RESULTS.md")
    print("Updated model_hub/pipeline-card/evaluation_results.json")
    print("Created BENCHMARK_RESULTS.md")


if __name__ == "__main__":
    main()
