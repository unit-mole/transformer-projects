# Validation Report — Portfolio-Scale Project 05 Upgrade

## Completed checks

- Expanded dataset rebuilt successfully.
- Expanded dataset size: 401 examples.
- Topic groups: 203.
- Splits: 323 training, 42 validation, and 36 test examples.
- Dataset structural validation issues: 0.
- Topic-group split leakage issues: 0.
- Python test suite: 11 tests passed.
- Python compilation: passed for application, source, scripts, and tests.
- End-to-end notebook: valid Jupyter JSON and all code cells parse as Python.
- Project JSON files: valid.
- Static frontend JavaScript syntax: passed.
- Dedicated GitHub Actions workflow: included.

## Checks that require the user's RTX environment

The generated package cannot claim trained-model scores until the notebook is run on the user's system. The following remain intentionally unexecuted in this source bundle:

- downloading FLAN-T5 and metric models;
- LoRA training on the NVIDIA RTX GPU;
- held-out base-model generation;
- held-out LoRA-adapter generation;
- BERTScore, ROUGE, semantic relevance, loss, perplexity, and latency calculation;
- paired bootstrap comparison;
- manual factuality review;
- Hugging Face adapter upload;
- merged ONNX export and browser-model publication.

The notebook saves each real result directly into JSON, CSV, Markdown, and PNG artifacts after execution.
