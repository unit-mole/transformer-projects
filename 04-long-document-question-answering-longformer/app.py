"""Hugging Face Spaces and local Gradio entry point."""

import os

from gradio_app import CUSTOM_CSS, demo


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        show_error=True,
        css=CUSTOM_CSS,
    )
