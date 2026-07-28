from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.qasper_training import PROFILES, recommend_profile


def main() -> None:
    result = recommend_profile()
    result["python"] = sys.version
    result["platform"] = platform.platform()
    result["available_profiles"] = {name: profile.to_dict() for name, profile in PROFILES.items()}
    try:
        import torch

        result["torch_version"] = torch.__version__
        result["torch_cuda_version"] = torch.version.cuda
        result["cudnn_version"] = torch.backends.cudnn.version()
    except Exception as exc:
        result["torch_error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
