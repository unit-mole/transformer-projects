from __future__ import annotations
import argparse
from src.model_conversion import export_with_optimum


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a Hugging Face checkpoint to ONNX.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="models/onnx_model")
    args = parser.parse_args()
    export_with_optimum(args.checkpoint, args.output)

if __name__ == "__main__":
    main()
