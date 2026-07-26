#!/usr/bin/env python
"""Train and save a LoRA adapter. Run in Colab, Kaggle, or a GPU environment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.model_training import train_lora


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_DIR / "configs/config.yaml"))
    args = parser.parse_args()
    print(json.dumps(train_lora(args.config), indent=2, default=str))


if __name__ == "__main__":
    main()
