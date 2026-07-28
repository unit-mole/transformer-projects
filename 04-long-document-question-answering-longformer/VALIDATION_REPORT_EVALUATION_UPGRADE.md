# Validation Report — Project 04 Evaluation Upgrade

## Validation completed in the artifact environment

- Python compilation passed for `src/`, `scripts/`, and tests.
- **18 Python unit tests passed**.
- **8 browser JavaScript tests passed**.
- JavaScript syntax validation passed.
- The complete evaluation notebook is valid notebook JSON and contains 29 cells.
- Placeholder JSON files are valid and intentionally marked `not_run`.
- The dedicated GitHub Actions workflow validates the notebook, new source
  modules, scripts, tests, configuration, and evaluation artifacts without
  downloading models or datasets.
- The incorrect article-normalization control characters in
  `src/model_evaluation.py` were replaced with the proper regular expression:
  `r"\b(a|an|the)\b"`.
- The project `.gitignore` keeps generated evaluation JSON, CSV, Markdown, and PNG artifacts trackable while excluding raw QASPER data, model weights, caches, and trainer checkpoints.

## What was not executed here

The artifact environment has no NVIDIA GPU and cannot download QASPER or model
weights. Therefore, it did not:

- download the official QASPER archives;
- fine-tune Longformer;
- run BERT or Longformer benchmark inference;
- create actual EM, F1, evidence, latency, or GPU-memory values;
- upload a model to Hugging Face.

No metrics were invented. The RTX notebook and command-line scripts perform
those steps on the user's system and replace the `not_run` artifacts with real
results.

## Required final validation on the RTX workstation

1. Confirm `python scripts/check_gpu.py` reports CUDA as available.
2. Run every cell in
   `notebooks/complete_longformer_training_evaluation_pipeline.ipynb`.
3. Confirm `outputs/evaluation_manifest.json` has `status: completed`.
4. Confirm the comparison includes at least 100 evaluated examples.
5. Review the generated weak and incorrect predictions manually.
6. Confirm the README and model card contain actual generated metrics.
7. Upload the fine-tuned checkpoint only after reviewing all results.
