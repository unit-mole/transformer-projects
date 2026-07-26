"""Upload the exported browser model directory to a Hugging Face model repo."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", default="models/browser_model")
    parser.add_argument("--repo-id", required=True, help="Example: username/ml-ds-flan-t5-small-onnx")
    parser.add_argument("--private", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    folder = Path(args.folder)
    if not folder.exists():
        raise FileNotFoundError(f"Browser model directory not found: {folder}")

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub from requirements-export.txt.") from exc

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(folder),
        commit_message="Upload merged ONNX model for Transformers.js",
    )
    print(f"Uploaded {folder} to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
