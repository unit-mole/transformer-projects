from __future__ import annotations
import argparse, json, sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.benchmark_utils import validate_outputs
parser = argparse.ArgumentParser()
parser.add_argument("--minimum-samples", type=int, default=200)
args = parser.parse_args()
print(json.dumps(validate_outputs(PROJECT_ROOT, args.minimum_samples), indent=2))
