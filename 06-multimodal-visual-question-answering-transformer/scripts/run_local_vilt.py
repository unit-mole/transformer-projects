from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from vqa.inference_pipeline import VQAInferencePipeline

def main() -> None:
    parser = argparse.ArgumentParser(description="Run local ViLT VQA inference.")
    parser.add_argument("image")
    parser.add_argument("question")
    parser.add_argument("--model-id", default="dandelin/vilt-b32-finetuned-vqa")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    pipeline = VQAInferencePipeline(model_id=args.model_id, device=args.device)
    print(json.dumps(pipeline.predict(args.image, args.question).to_dict(), indent=2))

if __name__ == "__main__":
    main()
