from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model_quantization import quantize_dynamic_onnx


def main() -> None:
    parser = argparse.ArgumentParser(description="Dynamically quantize an ONNX model to int8 weights.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(f"Quantized model: {quantize_dynamic_onnx(args.input, args.output)}")


if __name__ == "__main__":
    main()
