"""ONNX export helpers."""
from __future__ import annotations

import subprocess
from pathlib import Path


def export_with_optimum(checkpoint: str, output_dir: str | Path, task: str = "image-classification") -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    command = [
        "optimum-cli", "export", "onnx",
        "--model", checkpoint,
        "--task", task,
        str(output),
    ]
    subprocess.run(command, check=True)
    return output


def validate_export_directory(path: str | Path) -> list[Path]:
    base = Path(path)
    required = [base / "config.json", base / "preprocessor_config.json"]
    missing = [p for p in required if not p.exists()]
    onnx_files = list(base.rglob("*.onnx"))
    if missing or not onnx_files:
        details = ", ".join(str(p) for p in missing) or "no ONNX file"
        raise FileNotFoundError(f"Incomplete export: {details}")
    return onnx_files
