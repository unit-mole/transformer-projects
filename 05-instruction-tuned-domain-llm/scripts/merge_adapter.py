#!/usr/bin/env python
"""Merge a trained LoRA adapter into its FLAN-T5 base model and save it."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ModelConfig
from src.model_loader import load_model_and_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-id", default=os.getenv("ADAPTER_ID", ""))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "models" / "merged_model"))
    args = parser.parse_args()
    if not args.adapter_id:
        raise ValueError("Provide --adapter-id or set ADAPTER_ID.")

    config = ModelConfig(adapter_id=args.adapter_id)
    loaded = load_model_and_tokenizer(config, merge_adapter=True)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    loaded.model.save_pretrained(output, safe_serialization=True)
    loaded.tokenizer.save_pretrained(output)
    print(f"Merged model saved to {output}")


if __name__ == "__main__":
    main()
