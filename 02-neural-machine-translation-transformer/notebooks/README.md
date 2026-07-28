# Notebooks

- `neural_machine_translation_transformer.ipynb`: data inspection, language detection, and model-backed inference.
- `translation_evaluation_and_error_analysis.ipynb`: actual SacreBLEU, chrF, latency, plots, and manual-review export.
- `archive/`: original supplied notebook for traceability.

The production application imports code from `src/`; notebooks are not the production runtime.

## 03 — Portfolio-grade MarianMT fine-tuning and evaluation

`03_portfolio_grade_marianmt_finetuning_evaluation.ipynb` is the final recruiter-facing experiment notebook. It evaluates pretrained models, fine-tunes both directions on an IIT Bombay subset, evaluates the fine-tuned models on the same held-out test data, generates bootstrap comparisons, prepares manual error analysis, and writes all JSON/CSV/PNG artifacts used by the README and Static Space.
