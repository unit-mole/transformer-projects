#!/usr/bin/env python
"""Run the portfolio app locally."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from gradio_app import CSS, demo

if __name__ == "__main__":
    demo.launch(inbrowser=True, css=CSS)
