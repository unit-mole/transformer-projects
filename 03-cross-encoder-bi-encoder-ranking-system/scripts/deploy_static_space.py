from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload the Vite dist folder to a Hugging Face Static Space."
    )
    parser.add_argument(
        "--dist-dir",
        default="web/dist",
        help="Path to the built Vite directory.",
    )
    parser.add_argument(
        "--space-id",
        default=os.getenv("HF_SPACE_ID"),
        help="Hugging Face Space ID in owner/name format.",
    )
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is required.")
    if not args.space_id:
        raise SystemExit("HF_SPACE_ID or --space-id is required.")

    dist_dir = Path(args.dist_dir).resolve()
    if not dist_dir.exists():
        raise SystemExit(
            f"Build directory does not exist: {dist_dir}. Run npm run build."
        )

    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.space_id,
        repo_type="space",
        space_sdk="static",
        private=False,
        exist_ok=True,
    )
    api.upload_folder(
        repo_id=args.space_id,
        repo_type="space",
        folder_path=str(dist_dir),
        commit_message="Deploy Project 03 Static Space from GitHub Actions",
        delete_patterns=["*"],
    )

    print(f"Uploaded Static Space: https://huggingface.co/spaces/{args.space_id}")


if __name__ == "__main__":
    main()
