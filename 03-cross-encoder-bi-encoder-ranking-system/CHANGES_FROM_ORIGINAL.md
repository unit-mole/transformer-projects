# Changes From the Original DocRank360 Notebook

## Original foundation retained

The attached notebook established:

- MiniLM sentence embeddings;
- semantic retrieval;
- MS MARCO cross-encoder reranking;
- ranking latency analysis;
- quality-analytics positioning;
- synthetic-first demonstration data.

The original notebook remains under:

```text
notebooks/original_docrank360_notebook.ipynb
```

## Python productionization

The notebook was converted into:

- modular `src/` components;
- reusable preprocessing;
- query, document and qrels loaders;
- NumPy cosine index;
- candidate retrieval pipeline;
- cross-encoder reranking pipeline;
- two-stage ranking engine;
- Recall@K, MRR@10 and nDCG@10;
- latency benchmarking;
- Gradio local application;
- tests;
- Docker;
- GitHub Actions;
- model card;
- manual error-analysis framework.

## Browser deployment layer

A separate Vite project now exists under:

```text
web/
```

It adds:

- Transformers.js;
- ONNX Runtime Web;
- q8 browser models;
- model-loading progress;
- real browser document embeddings;
- cosine retrieval;
- browser cross-encoder scoring;
- live sample-query metrics;
- downloadable JSON;
- free Hugging Face Static Space deployment.

## Model Hub layer

The repository now includes:

```text
model_hub/pipeline-card/
```

This provides transparent system documentation without claiming ownership of
the base model weights.

Fine-tuned model-card templates are included but must not be published as
trained models until genuine fine-tuning or conversion has occurred.

## Important technical correction

The original weighted score:

```text
0.35 × bi-encoder score + 0.65 × cross-encoder score
```

was removed from the primary ranking pipeline. The two model outputs are on
different scales. The final system uses:

```text
bi-encoder → candidate generation
cross-encoder → candidate reranking
```

This is clearer and technically aligned with the stated two-stage architecture.

## Text processing correction

ASCII-only cleanup was replaced with Unicode-preserving NFKC normalization.
This prevents unnecessary damage to multilingual names, symbols and domain
terms.
