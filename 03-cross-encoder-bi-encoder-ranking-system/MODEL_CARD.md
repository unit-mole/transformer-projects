# Model Card — DocRank360 Two-Stage Ranking Pipeline

## System summary

DocRank360 is a two-stage information-retrieval system.

| Stage | Python model | Browser model |
|---|---|---|
| Candidate retrieval | `sentence-transformers/all-MiniLM-L6-v2` | `Xenova/all-MiniLM-L6-v2` |
| Candidate reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` | `Xenova/ms-marco-MiniLM-L-6-v2` |

## Portfolio contribution

This project contributes:

- modular Python retrieval and reranking code;
- preprocessing and dataset validation;
- NumPy vector indexing;
- ranking metrics;
- latency benchmarking;
- Gradio comparison application;
- Vite browser application;
- Transformers.js inference;
- Static Space deployment;
- tests and CI;
- pipeline documentation and responsible-use controls.

## Training status

- New bi-encoder fine-tuning: **No**
- New cross-encoder fine-tuning: **No**
- New browser conversion by this project: **No**
- Original and converted base models are credited above.

The project must not be presented as having trained those pretrained weights.

## Task

- semantic candidate retrieval;
- query-document relevance scoring;
- cross-encoder reranking;
- ranking evaluation;
- browser and Python latency analysis.

## Data

The public sample includes:

- 24 synthetic documents;
- 12 synthetic queries;
- 36 graded qrels;
- quality analytics, information retrieval, RAG and fictional job-search topics.

No private company data or personal resumes are included.

## Preprocessing

- HTML entity decoding;
- Unicode NFKC normalization;
- HTML tag removal;
- whitespace normalization;
- minimum-length validation;
- duplicate removal;
- title-document concatenation;
- graded relevance validation.

## Python inference

The Python implementation:

1. embeds documents with Sentence Transformers;
2. normalizes embeddings;
3. retrieves top candidates with cosine similarity;
4. scores query-document pairs with CrossEncoder;
5. reranks by cross-encoder score.

## Browser inference

The Static Space:

1. downloads q8 ONNX models;
2. generates document embeddings in browser memory;
3. embeds the query;
4. performs cosine retrieval;
5. tokenizes query-document pairs;
6. applies the sequence-classification reranker;
7. displays rank movement and metrics.

## Evaluation

Required metrics:

- Recall@5;
- Recall@10;
- bi-encoder MRR@10;
- reranked MRR@10;
- bi-encoder nDCG@10;
- reranked nDCG@10;
- MRR improvement;
- nDCG improvement;
- query embedding latency;
- candidate retrieval latency;
- cross-encoder reranking latency;
- total latency.

No placeholder value should be presented as a measured result.

## Intended use

- educational information-retrieval demonstration;
- portfolio review;
- semantic search over public-safe text;
- learning two-stage ranking;
- prototyping quality case and knowledge-base retrieval.

## Not intended use

- sole-basis hiring or rejection decisions;
- immigration, legal, compensation or promotion decisions;
- factual verification;
- confidential or personally identifiable text in a public Space;
- production use without security, monitoring, fairness and domain evaluation.

## Risks and limitations

- small synthetic data does not establish production performance;
- bi-encoder candidate misses cannot be repaired by reranking;
- cross-encoder reranking can introduce regressions;
- scores are not calibrated probabilities;
- browser speed varies by device;
- first load requires model downloads;
- lexical part numbers and codes may need hybrid search;
- pretrained models may contain occupational and language bias.

## Responsible use

Human review is required for consequential decisions. Do not upload private,
confidential, copyrighted, proprietary or personally identifiable text to the
public demo.

## Deployment

| Surface | Implementation |
|---|---|
| GitHub | Complete Python project |
| Hugging Face Model Hub | Pipeline card and evaluation documentation |
| Hugging Face Static Space | Vite + Transformers.js browser demo |
| Local comparison | Gradio |
