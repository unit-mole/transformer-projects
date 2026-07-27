from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.clip_model import ClipEncoder
from src.dataset_loader import load_gallery
from src.embedding_generator import save_browser_embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate normalized CLIP image embeddings for the browser gallery.")
    parser.add_argument("--model-id", default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    web_root = PROJECT_ROOT / "web"
    gallery = load_gallery(web_root / "data" / "image_gallery.json")
    image_paths = [web_root / str(item["image_path"]).removeprefix("./") for item in gallery]
    encoder = ClipEncoder(model_id=args.model_id, device=args.device)
    embeddings = encoder.encode_images(image_paths)
    output = save_browser_embeddings(
        [item["image_id"] for item in gallery],
        embeddings,
        web_root / "data" / "image_embeddings.json",
        model_id="Xenova/clip-vit-base-patch32",
    )
    print(f"Saved {len(gallery)} embeddings to {output}")


if __name__ == "__main__":
    main()
