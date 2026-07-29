# Validation Report — Large-Scale Evaluation Upgrade

## Completed in the build environment

- Python source compilation: passed
- Existing retrieval and reranking tests: passed
- New benchmark metric tests: passed
- New TF-IDF and BM25 baseline test: passed
- Total Python tests: **10 passed**
- Master benchmark notebook JSON: valid
- Notebook cells: **28**
- GitHub workflow updated for benchmark unit-test dependencies
- Downloaded benchmark corpora committed to Git: no
- Fine-tuned weights committed to Git: no

## Deliberately not claimed

The build environment could not access the external BEIR archive host or
Hugging Face model files. Therefore, it did not execute:

- the SciFact and NFCorpus downloads;
- MiniLM corpus embedding;
- MS MARCO cross-encoder reranking;
- GPU latency benchmarking;
- optional fine-tuning.

Those steps must run on the user's RTX system through:

```text
notebooks/04-large-scale-ranking-benchmark.ipynb
```

No benchmark score has been invented. Generated values should be pushed only
after the notebook completes and the outputs are reviewed.
