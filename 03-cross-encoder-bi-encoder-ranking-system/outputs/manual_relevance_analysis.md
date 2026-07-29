# Manual Relevance and Error Analysis

Complete this analysis after running the large-scale benchmark.

Primary source:

```text
outputs/benchmark/latest/ranking_examples.csv
```

## Improvement and regression review

| Dataset | Query ID | Query | Bi-encoder top result | Reranked top result | nDCG@10 before | nDCG@10 after | Delta | Human assessment |
|---|---|---|---|---|---:|---:|---:|---|
| `<dataset>` | `<id>` | `<query>` | `<title>` | `<title>` | `<value>` | `<value>` | `<value>` | `<observation>` |

## Candidate-retrieval failures

| Dataset | Query | Relevant document missed at K | Likely reason | Recommended improvement |
|---|---|---|---|---|
| `<dataset>` | `<query>` | `<document>` | `<lexical/domain/ambiguity>` | `<hybrid search/fine-tuning/etc.>` |

## Required examples

Record at least:

- five largest reranking improvements;
- five largest reranking regressions;
- three cases where TF-IDF or BM25 beats the bi-encoder;
- three cases where dense retrieval beats both lexical baselines;
- three queries where the relevant document is absent from the top 100;
- ambiguous scientific or biomedical queries;
- cross-encoder overconfidence or domain mismatch;
- latency outliers.

## Interpretation rules

- Do not describe a score as a probability.
- Do not hide negative reranking deltas.
- Distinguish candidate-recall failures from reranker failures.
- Report whether fine-tuning improves SciFact but harms NFCorpus transfer.
- Do not make claims beyond the evaluated datasets.
