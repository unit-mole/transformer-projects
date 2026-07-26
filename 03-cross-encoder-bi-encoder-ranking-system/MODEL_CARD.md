# Model Card — DocRank360 Ranking Pipeline

## Model details

| Component | Model |
|---|---|
| Bi-encoder | `sentence-transformers/all-MiniLM-L6-v2` |
| Cross-encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Index | Normalized NumPy embedding matrix |
| Similarity | Cosine similarity |
| Task | Candidate retrieval and query-document reranking |

This repository does not redistribute the pretrained weights. It downloads the
models from their Hugging Face repositories.

## Intended use

- educational demonstration of two-stage ranking;
- semantic search over a small public-safe corpus;
- learning retrieval metrics and latency tradeoffs;
- portfolio demonstration of Sentence Transformers and Gradio;
- prototyping quality-case, knowledge-base, and RAG retrieval workflows.

## Not intended use

- final hiring, rejection, promotion, compensation, immigration, or legal decisions;
- ranking people without appropriate governance and human review;
- factual verification;
- confidential or personally identifiable data in a public Space;
- production deployment without domain evaluation, security review, monitoring,
  fairness analysis, and data-governance controls.

## Data

The committed sample contains 24 synthetic documents, 12 synthetic queries, and
36 graded qrels. Topics include quality analytics, information retrieval, RAG,
evaluation, deployment, and fictional job descriptions.

The project can be adapted to MS MARCO or another licensed query-document
ranking dataset. Large or restricted datasets should not be committed.

## Preprocessing

- HTML entity decoding;
- Unicode NFKC normalization;
- HTML tag removal;
- whitespace normalization;
- missing and duplicate removal;
- title-document concatenation;
- relevance values restricted to 0–3.

The same cleaning functions are used during indexing and inference.

## Training

No training or fine-tuning is performed by this project. It uses pretrained
models. Future versions can fine-tune the bi-encoder with hard negatives or
train a domain reranker.

## Evaluation

Required outputs:

- Recall@5 and Recall@10;
- bi-encoder and reranked MRR@10;
- bi-encoder and reranked nDCG@10;
- MRR and nDCG improvement;
- query embedding, retrieval, reranking, and total latency;
- query-level ranking examples;
- manual relevance and failure analysis.

Committed placeholders contain `status: not_run`; run
`python scripts/evaluate_model.py` to generate actual values.

## Risks and limitations

- small synthetic data does not represent production traffic;
- lexical entities, part numbers, and rare terms may benefit from hybrid search;
- the bi-encoder can miss relevant candidates;
- the cross-encoder can be overconfident or reorder results incorrectly;
- scores are not calibrated probabilities;
- pretrained data can encode social and occupational bias;
- CPU cold start includes model download and loading;
- rankings must be reviewed by a human in consequential settings.

## Inference example

```python
from src.ranking_engine import TwoStageRankingEngine
from src.settings import Settings

engine = TwoStageRankingEngine.from_settings(Settings.from_yaml())
response = engine.search(
    "How can I find similar quality complaints?",
    candidate_k=10,
    rerank_k=5,
)
print(response.reranked_results)
```

## Deployment

- Platform: Hugging Face Spaces
- SDK: Gradio
- Entry point: `app.py`
- Hardware target: CPU
- Training at startup: no
- Full-scale indexing at startup: no
- Small sample index fallback: yes
