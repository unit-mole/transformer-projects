from __future__ import annotations

import os

from gradio_app import build_demo

demo = build_demo().queue(default_concurrency_limit=2)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        show_error=True,
    )
