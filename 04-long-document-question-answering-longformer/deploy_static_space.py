from __future__ import annotations

from pathlib import Path
import sys

from huggingface_hub import HfApi, whoami

REPO_ID = "anmol-unitmole/long-document-question-answering-longformer"
ROOT = Path(__file__).resolve().parent
SPACE_DIR = ROOT / "hf-space-ready"


def main() -> int:
    if not SPACE_DIR.is_dir():
        print(f"ERROR: Missing deployment folder: {SPACE_DIR}")
        return 1

    try:
        account = whoami()
    except Exception as exc:
        print("ERROR: Hugging Face authentication was not found.")
        print("Run: hf auth login")
        print(exc)
        return 1

    username = account.get("name") or account.get("fullname") or "unknown"
    print(f"Authenticated as: {username}")
    print(f"Deploying prebuilt static files to: {REPO_ID}")
    print("No Hugging Face build command or credits are required.")

    api = HfApi()
    result = api.upload_folder(
        folder_path=str(SPACE_DIR),
        repo_id=REPO_ID,
        repo_type="space",
        delete_patterns="*",
        commit_message="Deploy final prebuilt credit-free Project 04 Static Space",
    )
    print("Deployment completed successfully.")
    print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
