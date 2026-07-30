# Model Card — AI Portfolio RAG Assistant

## System name

**AI Portfolio RAG Assistant — Transformer Project 10**

## System task

Source-cited retrieval-augmented question answering over verified public machine-learning portfolio documentation.

## Components

| Component | Model / approach | Role |
|---|---|---|
| Primary dense retriever | `sentence-transformers/all-MiniLM-L6-v2` | 384-dimensional semantic document/query embeddings |
| Alternative retriever | `intfloat/e5-small-v2` | Query/passage-prefixed dense retrieval comparison |
| Optional reranker | `cross-encoder/ms-marco-MiniLM-L6-v2` | Reranks dense candidates |
| Local instruction generator | `google/flan-t5-base` | GPU evaluation of grounded answer generation |
| Deployed generator | Configurable Hugging Face instruction model | Server-side answer generation on Vercel |
| Safe fallback | Deterministic grounded extractive composer | Refuses weak evidence and works without generation API |
| Groundedness evaluator | `cross-encoder/nli-deberta-v3-small` | Claim/evidence entailment scoring |

## Transformer status

The final system is considered Transformer-powered only when `public/data/metadata.json` reports real MiniLM/E5 embeddings and the deployed query encoder uses the same model. The starter hash vectors are a functional fallback, not the final model.

## Intended use

- Recruiters and technical reviewers exploring documented projects.
- Portfolio visitors asking about models, datasets, metrics, skills, and deployments.
- Educational demonstration of RAG, semantic search, evaluation, source attribution, full-stack AI development, and Vercel deployment.

## Not intended use

- Official resume or employment verification.
- Legal, HR, or professional-reference decisions.
- Search over confidential company documents in this public repository.
- Claims about skills, metrics, employers, or results that are not present in indexed sources.

## Retrieval method

1. Public Markdown is cleaned without removing technical terminology.
2. Documents are split into section-aware overlapping chunks.
3. Normalized document embeddings are generated offline.
4. Runtime queries are embedded using the same model and prefix configuration.
5. Cosine similarity is combined with lexical coverage.
6. Optional filters and cross-encoder reranking refine results.
7. Low-confidence evidence can trigger refusal.

## Generation method

The instruction model receives only the retrieved chunks and a strict prompt requiring source citations after factual claims. The grounded extractive fallback remains available when hosted generation fails or is disabled.

## Citation method

Each retrieved chunk is assigned an `[S#]` ID. The UI displays project, source file, heading, chunk ID, evidence, similarity score, path, and repository URL.

## Evaluation protocol

### Retrieval

- Hit Rate@K
- Precision@K
- Recall@K
- Mean Reciprocal Rank
- Mean Average Precision@K
- nDCG@K

### Answers

- Claim-level NLI groundedness
- Citation precision
- Citation completeness
- Unsupported-claim rate
- Unsupported-question refusal accuracy
- Manual error analysis

### Performance

- Query embedding latency
- Retrieval latency
- Generation latency
- Total local latency
- Median, P90, P95, minimum, and maximum
- Deployed Vercel wall-clock latency

Metrics must be regenerated after corpus, chunking, model, or prompt changes. No result should be copied from another run or estimated manually.

## Risks and limitations

- An incomplete corpus produces incomplete answers.
- NLI-based groundedness is an automated proxy and may misclassify nuanced claims.
- Citation presence does not alone prove citation correctness.
- Similarity thresholds require validation against supported and unsupported questions.
- Hosted model availability and latency may change.
- Public documents can become stale.
- Multiple similar projects can confuse retrieval unless metadata and evaluation labels are accurate.

## Bias and privacy

The corpus reflects only selected public portfolio content and therefore does not represent all experience. Private employment records, company data, GCS cases, quality records, emails, customer/supplier information, credentials, proprietary files, and PII must not be included.

## Deployment

The Next.js application is deployed on Vercel. Python, PyTorch, and local models are used offline only; runtime vector artifacts are static JSON. API tokens remain server-side.

## Responsible-use statement

Treat every answer as a source-cited portfolio summary requiring human review—not as an official statement of employment, qualifications, or performance.
