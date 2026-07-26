"""Hugging Face Spaces entry point."""

import os

from gradio_app import CSS, demo

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        css=CSS,
    )
