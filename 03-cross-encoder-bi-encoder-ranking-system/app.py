"""Hugging Face Spaces entry point."""

from gradio_app import build_demo

demo = build_demo()

if __name__ == "__main__":
    demo.launch()
