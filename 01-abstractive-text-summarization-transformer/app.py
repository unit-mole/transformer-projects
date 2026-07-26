from __future__ import annotations

import os

from gradio_app import build_demo

# Hugging Face Spaces imports this module and discovers `demo`.
demo = build_demo()

if __name__ == "__main__":
    demo.queue().launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        show_error=True,
    )
