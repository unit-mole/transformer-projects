from __future__ import annotations
import argparse
from pathlib import Path
from PIL import Image
from src.attention_visualization import generate_attention_visualization
from src.vit_model import load_vit_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a genuine attention-rollout example.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="outputs/attention_visualization_examples.png")
    args = parser.parse_args()
    if not Path(args.image).exists():
        raise FileNotFoundError(args.image)
    processor, model = load_vit_model(args.checkpoint)
    model.eval()
    generate_attention_visualization(model, processor, Image.open(args.image), args.output)
    print(args.output)

if __name__ == "__main__":
    main()
