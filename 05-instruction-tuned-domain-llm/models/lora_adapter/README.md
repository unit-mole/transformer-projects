# LoRA Adapter Directory

This repository does not fabricate or ship untrained adapter weights. Run `notebooks/05_full_training_evaluation_pipeline.ipynb`, complete dataset and response review, and promote the experiment. The reviewed local directory will then contain `adapter_config.json` and `adapter_model.safetensors`.

For Hugging Face deployment, publish the adapter to a separate model repository and set `ADAPTER_ID`. The Gradio app must report `lora_adapter`; `base_model_fallback` means the trained adapter was not loaded.
