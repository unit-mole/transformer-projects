#!/usr/bin/env python
"""Launch the Gradio demo locally."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from gradio_app import APP_CSS, APP_THEME, demo

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch(show_error=True, theme=APP_THEME, css=APP_CSS)
