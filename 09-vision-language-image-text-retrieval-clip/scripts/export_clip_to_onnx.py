from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model_conversion import export_clip_onnx


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a CLIP checkpoint to ONNX using Optimum CLI.")
    parser.add_argument("--model-id", default="openai/clip-vit-base-patch32")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "models" / "onnx_export")
    args = parser.parse_args()
    print(f"Exported to {export_clip_onnx(args.model_id, args.output_dir)}")


if __name__ == "__main__":
    main()
