from __future__ import annotations

from pathlib import Path


def quantize_dynamic_onnx(input_path: str | Path, output_path: str | Path) -> Path:
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as exc:
        raise RuntimeError("onnxruntime is required; install requirements-model.txt") from exc
    source = Path(input_path)
    target = Path(output_path)
    if not source.exists():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(str(source), str(target), weight_type=QuantType.QInt8)
    return target
