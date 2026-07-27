from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    required = [
        "index.html", "style.css", "app.js", "clip_preprocessing.js", "clip_inference.js",
        "retrieval.js", "zero_shot.js", "metadata.json", "zero_shot_labels.json",
        "data/image_gallery.json", "data/image_embeddings.json", "data/captions.json",
    ]
    web = PROJECT_ROOT / "web"
    missing = [item for item in required if not (web / item).exists()]
    if missing:
        raise FileNotFoundError(f"Missing browser assets: {missing}")
    manifest = {"valid": True, "asset_count": len(required), "assets": required}
    (PROJECT_ROOT / "outputs" / "browser_asset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Browser assets are ready for GitHub Pages.")


if __name__ == "__main__":
    main()
