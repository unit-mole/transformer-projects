from __future__ import annotations
import json, sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.benchmark_utils import hardware_report
report = hardware_report()
print(json.dumps(report, indent=2))
if not report.get("cuda_available"):
    raise SystemExit("CUDA is not available to PyTorch. Install a CUDA-enabled PyTorch build first.")
print("GPU validation passed.")
