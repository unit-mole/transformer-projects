# Validation Report

## Completed checks

- Python source compilation: **passed**
- Gradio application import: **passed**
- Dataset loading: **passed**
- Documents loaded: **24**
- Queries loaded: **12**
- Graded qrels loaded: **36**
- Pytest suite: **7 passed**
- Tests download pretrained models: **no**
- Application trains a model at startup: **no**

## Deliberately not claimed

The execution environment used to build this downloadable bundle did not have
the `sentence-transformers` package or internet model access. Therefore, the
actual MiniLM bi-encoder and MS MARCO cross-encoder were not executed here.

The evaluation JSON files correctly remain marked `status: not_run`. Install
`requirements.txt` and run:

```bash
python scripts/build_index.py
python scripts/evaluate_model.py
python scripts/benchmark_latency.py
```

Only those measured results should be displayed in the GitHub README or demo.
