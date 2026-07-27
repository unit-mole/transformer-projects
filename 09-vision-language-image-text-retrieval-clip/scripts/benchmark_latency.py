from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.clip_model import ClipEncoder
from src.latency_benchmark import benchmark


def main() -> None:
    encoder = ClipEncoder()
    result = benchmark(encoder.encode_text, ["a red car on a road"], repeats=5)
    payload = {"status": "measured", "text_encoding": result, "note": "Python CPU benchmark; browser latency is displayed live in the deployed app."}
    target = PROJECT_ROOT / "outputs" / "latency_results.json"
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
