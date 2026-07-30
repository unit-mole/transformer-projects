"""Hugging Face Spaces entry point. Training is intentionally excluded."""
from __future__ import annotations

import os

from gradio_app import APP_CSS, APP_THEME, demo


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        show_error=True,
        theme=APP_THEME,
        css=APP_CSS,
    )
