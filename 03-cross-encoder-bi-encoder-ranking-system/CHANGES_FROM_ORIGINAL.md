# Changes From the Attached DocRank360 Notebook

The attached notebook was carefully used as the base concept. Its strongest
elements were retained:

- synthetic-first runnable corpus;
- `all-MiniLM-L6-v2` bi-encoder;
- `ms-marco-MiniLM-L-6-v2` cross-encoder;
- two-stage retrieval concept;
- fallback-aware thinking;
- latency measurement;
- exportable analysis;
- connection to quality analytics.

## Productionization changes

| Original notebook | Portfolio-ready project |
|---|---|
| One 124-cell notebook | Modular `src/`, `scripts/`, `tests/`, and two focused notebooks |
| Streamlit application | Gradio application for Hugging Face Spaces |
| Synthetic plus SQuAD QA corpus | Explicit query-document-qrels ranking sample |
| Hit@K and MRR | Recall@K, MRR@10, graded nDCG@10, and improvement deltas |
| One combined search latency | Query embedding, retrieval, reranking, total, median, and p95 |
| Weighted `0.35 × bi + 0.65 × cross` final score | Cross-encoder-only ordering within reranked candidates |
| ASCII-only cleanup | Unicode-preserving NFKC cleanup |
| Lexical fallback could look like the final model | Tests use deterministic fakes; public metrics remain `not_run` until real models execute |
| Streamlit code embedded as a long string | Maintainable `app.py` and `gradio_app.py` |
| Optional public dataset fetched in notebook | Safe committed sample; external benchmark instructions documented |
| Limited rank-change visibility | Retrieval rank, reranked rank, movement, and reranked flag |
| No repository CI | Lightweight pytest and import validation workflow |
| No Space card | Hugging Face YAML metadata and deployment guide |
| No formal model card | Intended use, prohibited use, risks, data, metrics, and deployment details |

## Why the weighted score was removed

A weighted combination can be useful after score calibration and validation, but
the original bi-encoder cosine scores and cross-encoder logits are on different
scales. Combining them with fixed weights can be misleading. This project uses
the bi-encoder strictly for candidate generation and the cross-encoder strictly
for reranking, which matches the stated two-stage architecture.

## Original artifact

The uploaded notebook is retained as:

`notebooks/original_docrank360_notebook.ipynb`

This preserves the starting point while keeping the deployable application
separate and maintainable.
