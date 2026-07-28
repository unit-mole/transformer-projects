from __future__ import annotations

import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload the project-fine-tuned Longformer checkpoint.")
    parser.add_argument("--repo-id", required=True, help="Example: unit-mole/longformer-qasper-document-qa")
    parser.add_argument("--private", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = PROJECT_ROOT / "models" / "qasper-longformer"
    if not (model_dir / "config.json").exists():
        raise FileNotFoundError(
            "No fine-tuned model was found. Run the complete notebook or fine_tune_longformer_qasper.py first."
        )
    card = PROJECT_ROOT / "MODEL_CARD_QASPER_FINETUNED.md"
    if card.exists():
        shutil.copy2(card, model_dir / "README.md")

    from huggingface_hub import HfApi
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer

    model = AutoModelForQuestionAnswering.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model.push_to_hub(args.repo_id, private=args.private)
    tokenizer.push_to_hub(args.repo_id, private=args.private)

    api = HfApi()
    for artifact in [
        PROJECT_ROOT / "outputs" / "baseline_comparison.json",
        PROJECT_ROOT / "outputs" / "controlled_context_length_comparison.json",
        PROJECT_ROOT / "outputs" / "EVALUATION_REPORT.md",
        PROJECT_ROOT / "outputs" / "training_summary.json",
    ]:
        if artifact.exists():
            api.upload_file(
                path_or_fileobj=str(artifact),
                path_in_repo=f"evaluation/{artifact.name}",
                repo_id=args.repo_id,
                repo_type="model",
            )
    print(f"Uploaded model and evaluation artifacts to {args.repo_id}")


if __name__ == "__main__":
    main()
