"""Export a merged FLAN-T5 checkpoint to a Transformers.js model repository.

The script uses Optimum's command-line exporter, stores ONNX files in an
`onnx/` subfolder, copies tokenizer/configuration files to the repository root,
and optionally creates dynamic INT8 (`*_quantized.onnx`) variants.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT_FILES = {
    "config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer_config.json",
    "added_tokens.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/merged_model")
    parser.add_argument("--output", default="models/browser_model")
    parser.add_argument(
        "--task",
        default="text2text-generation-with-past",
        help="Optimum ONNX task. Use text2text-generation if your Optimum version does not support with-past.",
    )
    parser.add_argument("--quantize", action="store_true")
    return parser.parse_args()


def run_export(model_path: Path, onnx_path: Path, task: str) -> None:
    command = [
        "optimum-cli",
        "export",
        "onnx",
        "--model",
        str(model_path),
        "--task",
        task,
        str(onnx_path),
    ]
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "`optimum-cli` was not found. Install `requirements-export.txt`."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "ONNX export failed. Review the Optimum output. If the selected task is "
            "unsupported, retry with `--task text2text-generation`."
        ) from exc


def quantize_models(onnx_path: Path) -> list[str]:
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as exc:
        raise RuntimeError("Install onnxruntime before using --quantize.") from exc

    outputs: list[str] = []
    for model_file in sorted(onnx_path.glob("*.onnx")):
        if any(tag in model_file.stem for tag in ("_quantized", "_int8", "_uint8")):
            continue
        quantized = model_file.with_name(f"{model_file.stem}_quantized.onnx")
        quantize_dynamic(
            model_input=str(model_file),
            model_output=str(quantized),
            weight_type=QuantType.QInt8,
        )
        outputs.append(quantized.name)
    return outputs


def copy_root_files(model_path: Path, output_path: Path) -> None:
    for source in model_path.iterdir():
        if source.is_file() and source.name in ROOT_FILES:
            shutil.copy2(source, output_path / source.name)


def write_model_card(output_path: Path, source_model: str, quantized: list[str]) -> None:
    card = f"""---
library_name: transformers.js
base_model: google/flan-t5-small
pipeline_tag: text2text-generation
license: apache-2.0
---

# ML/Data Science Instruction-Tuned FLAN-T5 — ONNX

This repository contains a merged and ONNX-exported version of the Project 05
ML/Data Science Learning Assistant.

- **Source merged checkpoint:** `{source_model}`
- **Architecture:** FLAN-T5 encoder-decoder Transformer
- **Fine-tuning method:** LoRA / PEFT before adapter merging
- **Browser runtime:** Transformers.js + ONNX Runtime Web
- **Quantized files generated:** {len(quantized)}

Publish this repository only after training and evaluating the adapter. Replace
this template with actual dataset, training, metric, limitation, and responsible-use details.
"""
    (output_path / "README.md").write_text(card, encoding="utf-8")


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    output_path = Path(args.output)
    onnx_path = output_path / "onnx"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Merged model not found at '{model_path}'. Run merge_lora_adapter.py first."
        )

    if output_path.exists():
        shutil.rmtree(output_path)
    onnx_path.mkdir(parents=True, exist_ok=True)

    run_export(model_path, onnx_path, args.task)
    copy_root_files(model_path, output_path)
    quantized = quantize_models(onnx_path) if args.quantize else []
    write_model_card(output_path, str(model_path), quantized)

    metadata = {
        "source_model": str(model_path),
        "output_repository": str(output_path),
        "onnx_directory": str(onnx_path),
        "task": args.task,
        "quantized_files": quantized,
        "transformers_js_layout": True,
    }
    (output_path / "browser_export_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
