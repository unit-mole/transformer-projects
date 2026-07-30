from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter, sleep
import urllib.request

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def post_json(url: str, payload: dict, timeout: int) -> tuple[dict, float]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data, (perf_counter() - start) * 1000


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the deployed Vercel /api/chat endpoint.")
    parser.add_argument("--base-url", required=True, help="Example: https://your-app.vercel.app")
    parser.add_argument("--questions", type=Path, default=ROOT / "data/processed/evaluation_questions.json")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/deployed_latency_results.json")
    args = parser.parse_args()

    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    url = args.base_url.rstrip("/") + "/api/chat"
    records = []

    for repetition in range(args.repetitions):
        for item in questions:
            response, wall_ms = post_json(
                url,
                {"question": item["question"], "topK": args.top_k, "filters": {}},
                timeout=args.timeout,
            )
            metrics = response.get("metrics", {})
            records.append({
                "repetition": repetition + 1,
                "question_id": item["id"],
                "wall_clock_ms": round(wall_ms, 3),
                "embedding_ms": metrics.get("embeddingMs"),
                "retrieval_ms": metrics.get("retrievalMs"),
                "generation_ms": metrics.get("generationMs"),
                "server_total_ms": metrics.get("totalMs"),
                "retrieval_mode": response.get("runtime", {}).get("retrievalMode"),
                "generation_mode": response.get("runtime", {}).get("generationMode"),
                "warnings": response.get("warnings", []),
            })
            sleep(args.sleep)

    values = [row["wall_clock_ms"] for row in records]
    payload = {
        "status": "measured_on_deployed_vercel_app",
        "base_url": args.base_url,
        "summary": {
            "request_count": len(values),
            "mean_ms": round(float(np.mean(values)), 3),
            "median_ms": round(float(np.median(values)), 3),
            "p90_ms": round(float(np.percentile(values, 90)), 3),
            "p95_ms": round(float(np.percentile(values, 95)), 3),
            "min_ms": round(float(np.min(values)), 3),
            "max_ms": round(float(np.max(values)), 3),
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
