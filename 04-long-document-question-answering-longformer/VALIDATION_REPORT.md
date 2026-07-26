# Validation Report

Validation completed for the generated project package.

## Passed checks

- Python compilation: `app.py`, `gradio_app.py`, `src/`, `scripts/`, and `tests/`
- Unit tests: **10 passed**
- Gradio application import with model loading disabled
- Document loading tests for valid, empty, and unsupported files
- Paragraph-offset and overlapping word-chunk tests
- Answer-span extraction test
- Evidence-highlighting tests
- End-to-end inference-pipeline test with a deterministic fake model
- Gradio callback test with a deterministic fake model

## Not executed in this environment

The published Longformer checkpoint was not downloaded or benchmarked because
the artifact-building runtime does not provide internet access. Therefore:

- no real Longformer inference metric is claimed;
- `outputs/model_metrics.json` remains `status: not_run`;
- actual Exact Match, F1, evidence recall, latency, and context-length results
  must be generated locally or in the Hugging Face Space.

Run:

```bash
python scripts/evaluate_model.py
python scripts/run_context_analysis.py
```

before adding metric values to the README.
