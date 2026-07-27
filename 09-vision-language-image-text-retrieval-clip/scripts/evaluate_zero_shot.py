from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.clip_model import ClipEncoder
from src.text_preprocessing import create_label_prompts, parse_candidate_labels
from src.zero_shot_classifier import classify_from_embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a measured zero-shot example for one local image.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--labels", default="dog,cat,car,truck,bicycle,mountain,beach,forest,food,flower")
    args = parser.parse_args()

    labels = parse_candidate_labels(args.labels)
    encoder = ClipEncoder()
    image_embedding = encoder.encode_images([args.image])[0]
    label_embeddings = encoder.encode_text(create_label_prompts(labels))
    predictions = classify_from_embeddings(image_embedding, label_embeddings, labels)
    payload = {"status": "measured", "image": str(args.image), "predictions": [p.__dict__ for p in predictions]}
    target = PROJECT_ROOT / "outputs" / "zero_shot_classification_examples.json"
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
