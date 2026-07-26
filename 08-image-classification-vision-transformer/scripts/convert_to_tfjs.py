from __future__ import annotations
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Document optional TensorFlow.js conversion.")
    parser.add_argument("--saved-model", required=True)
    parser.add_argument("--output", default="models/tfjs_model")
    args = parser.parse_args()
    raise SystemExit(
        "This project uses ONNX/Transformers.js by default. For a TensorFlow SavedModel, install tensorflowjs and run: "
        f"tensorflowjs_converter --input_format=tf_saved_model {args.saved_model} {args.output}"
    )

if __name__ == "__main__":
    main()
