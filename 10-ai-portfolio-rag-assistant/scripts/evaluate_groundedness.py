from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    output = ROOT / "outputs/answer_groundedness_results.json"
    payload = {
        "status": "pending_manual_review",
        "metric": "answer_groundedness",
        "instructions": "Export app answers, review each claim against retrieved chunks, assign a score only after evidence review.",
        "records": []
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Created groundedness review template: {output}")

if __name__ == "__main__":
    main()
