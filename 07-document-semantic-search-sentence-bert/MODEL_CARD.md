# Model Card — Document Semantic Search Sentence-BERT

## Model details

| Field | Value |
|---|---|
| Task | Document semantic search / passage retrieval |
| Python embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Browser model | `Xenova/all-MiniLM-L6-v2` through Transformers.js |
| Architecture | MiniLM sentence-transformer bi-encoder |
| Embedding dimension | 384 |
| Pooling | Mean pooling |
| Normalization | L2 normalization |
| Similarity metric | Cosine similarity |
| Deployment | Static GitHub Pages application |

## Intended use

- Search public ML portfolio READMEs, model cards, dataset cards, and technical notes.
- Demonstrate browser-based Transformer inference and vector ranking.
- Explore semantic retrieval quality with Recall@K, MRR, similarity analysis, and latency benchmarking.
- Prototype a safe precursor to knowledge-base search and RAG.

## Not intended for

- Search over private, confidential, proprietary, personal, or regulated records in a public site.
- High-stakes decisions without human review.
- Treating cosine similarity as a calibrated probability.
- Production enterprise search without access controls, monitoring, governance, and retrieval evaluation.

## Evaluation

A completed offline run against eight labelled queries and the included 34-chunk corpus produced Recall@1 of **0.875**, Recall@3/5/10 of **1.000**, and MRR of **0.9375**. The Top-5 Python end-to-end latency benchmark averaged **4.22 ms** across 24 measurements. The false-positive and similarity-distribution review remains explicitly marked as pending; no cosine-analysis statistic is fabricated.

## Limitations and risks

- Similar wording can receive a high score even when the passage is not relevant to the user’s intent.
- Domain terminology not represented during model training may be embedded imperfectly.
- Chunk boundaries influence retrieval quality.
- Browser performance varies by device, browser, and model cache state.
- Publicly deployed corpora can expose any text committed to the repository.

## Bias and responsible use

Embedding models can reflect biases in their training data. Review false positives, missed results, and ranking behavior across topics. Do not place private company records, complaint details, employee data, emails, personally identifiable information, or copyrighted documents in the public corpus. Retrieved text must be reviewed by a human before real-world use.

## Deployment notes

The browser application either loads precomputed normalized document vectors or generates them locally using the browser-compatible ONNX model. Queries are embedded with the same model, and ranking is performed entirely in JavaScript. No search text is sent to an application backend.
