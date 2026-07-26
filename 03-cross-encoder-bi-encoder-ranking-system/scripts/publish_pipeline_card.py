from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CARD_FOLDER = PROJECT_ROOT / "model_hub" / "pipeline-card"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Publish the two-stage pipeline documentation repository. "
            "This does not upload or claim ownership of pretrained model weights."
        )
    )
    parser.add_argument(
        "--repo-id",
        default=os.getenv(
            "HF_MODEL_CARD_REPO_ID",
            "anmol-unitmole/docrank360-ranking-pipeline-card",
        ),
    )
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is required.")

    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=False,
        exist_ok=True,
    )
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(CARD_FOLDER),
        commit_message="Publish DocRank360 pipeline model card",
    )

    print(f"Published model card repository: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
