"""Merge a trained PEFT LoRA adapter into its FLAN-T5 base model.

The merged checkpoint is required before exporting a single browser-compatible
ONNX model. This script never trains a model and never invents adapter files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="google/flan-t5-small")
    parser.add_argument("--adapter", default="models/lora_adapter")
    parser.add_argument("--output", default="models/merged_model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter_path = Path(args.adapter)
    output_path = Path(args.output)

    if not adapter_path.exists():
        raise FileNotFoundError(
            f"LoRA adapter not found at '{adapter_path}'. Train the adapter first "
            "with `python scripts/train_lora.py`."
        )

    try:
        from peft import PeftModel
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Install export dependencies with `pip install -r requirements-export.txt`."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_path))
    merged_model = peft_model.merge_and_unload()

    output_path.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)

    metadata = {
        "base_model": args.base_model,
        "adapter": str(adapter_path),
        "merged_output": str(output_path),
        "fine_tuning_method": "LoRA / PEFT",
        "status": "merged",
    }
    (output_path / "merge_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
