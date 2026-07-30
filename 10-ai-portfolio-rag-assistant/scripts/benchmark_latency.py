from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    output = ROOT / "outputs/response_latency_results.json"
    payload = {
        "status": "run_after_deployment",
        "metric": "response_latency",
        "requiredFields": ["query_embedding_ms", "retrieval_ms", "generation_ms", "total_ms", "top_k", "corpus_size", "generator_mode"],
        "records": [],
        "note": "Collect repeated measurements against the deployed Vercel API. Do not treat local development latency as production latency."
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Created latency benchmark template: {output}")

if __name__ == "__main__":
    main()
