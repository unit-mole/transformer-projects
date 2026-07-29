from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    payload = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        payload.update(
            {
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_count": torch.cuda.device_count(),
                "gpu_total_memory_gb": round(properties.total_memory / 1024**3, 3),
                "compute_capability": f"{properties.major}.{properties.minor}",
                "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            }
        )
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
        payload["nvidia_smi"] = result.stdout.strip() or result.stderr.strip()
    except FileNotFoundError:
        payload["nvidia_smi"] = "nvidia-smi was not found on PATH"

    output = PROJECT_ROOT / "outputs" / "hardware" / "gpu_info.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available in PyTorch. Install a CUDA-enabled PyTorch build before the full benchmark."
        )


if __name__ == "__main__":
    main()
