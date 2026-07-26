from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vqa.inference_pipeline import VQAInferencePipeline
from vqa.latency import benchmark

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=str(ROOT / "data/sample_images/shapes_scene.png"))
    parser.add_argument("--question", default="What color is the circle?")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    pipeline = VQAInferencePipeline(device=args.device)
    result = benchmark(lambda: pipeline.predict(args.image, args.question), repeats=args.repeats)
    path = ROOT / "outputs/latency_results.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
