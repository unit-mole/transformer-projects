# Model artifacts

The application loads
`valhalla/longformer-base-4096-finetuned-squadv1` directly from the Hugging Face
Hub. This repository does **not** commit large model weights.

To use a separately fine-tuned checkpoint:

1. Save it under `models/long_document_qa_model/` and its tokenizer under
   `models/tokenizer/`, or upload it to a Hugging Face model repository.
2. Set `LONGDOCQA_MODEL_ID` to the local directory or Hub repository ID.
3. Update `model_metadata.json` and `MODEL_CARD.md`.
4. Never claim this project fine-tuned a checkpoint unless training was actually
   completed and documented.

Use Git LFS or the Hugging Face Hub for large `.safetensors` files.
