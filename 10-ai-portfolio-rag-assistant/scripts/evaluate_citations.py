from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    output = ROOT / "outputs/citation_correctness_results.json"
    payload = {
        "status": "pending_manual_review",
        "metric": "citation_correctness",
        "instructions": "Verify that each citation supports the preceding claim and references the correct project, file, section, and chunk.",
        "records": []
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Created citation review template: {output}")

if __name__ == "__main__":
    main()
