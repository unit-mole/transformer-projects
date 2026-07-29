# Project 03 Benchmark Results

> Generated from actual model execution. Do not edit metric values manually.

## Datasets

- **scifact (test):** 5,183 documents, 300 evaluated queries and 339 relevance judgments.
- **nfcorpus (test):** 3,633 documents, 323 evaluated queries and 12,334 relevance judgments.

## Model and baseline comparison

| dataset   | approach                      |   recall_at_10 |   mrr_at_10 |   ndcg_at_10 |   map_at_100 |   mean_query_ms |
|:----------|:------------------------------|---------------:|------------:|-------------:|-------------:|----------------:|
| scifact   | tfidf                         |         0.763  |      0.5843 |       0.6209 |       0.577  |          0.1513 |
| scifact   | bm25                          |         0.7809 |      0.6283 |       0.6613 |       0.6231 |          0.9793 |
| scifact   | bi_encoder                    |         0.7883 |      0.6068 |       0.6484 |       0.6055 |          0.2354 |
| scifact   | bi_encoder_plus_cross_encoder |         0.8089 |      0.6559 |       0.6868 |       0.6481 |         41.9802 |
| nfcorpus  | tfidf                         |         0.1447 |      0.4823 |       0.2933 |       0.1385 |          0.0766 |
| nfcorpus  | bm25                          |         0.1454 |      0.5161 |       0.3072 |       0.1457 |          0.4683 |
| nfcorpus  | bi_encoder                    |         0.1589 |      0.5088 |       0.319  |       0.1537 |          0.3376 |
| nfcorpus  | bi_encoder_plus_cross_encoder |         0.1594 |      0.5634 |       0.3453 |       0.1757 |         42.1207 |

## Paired bootstrap reranking analysis

### scifact

- **mrr_at_10:** mean delta `0.0491`, 95% CI `[0.0096, 0.0858]`, P(delta > 0) `0.994`.
- **ndcg_at_10:** mean delta `0.0384`, 95% CI `[0.0051, 0.0710]`, P(delta > 0) `0.986`.
- **map_at_100:** mean delta `0.0426`, 95% CI `[0.0052, 0.0774]`, P(delta > 0) `0.989`.

### nfcorpus

- **mrr_at_10:** mean delta `0.0546`, 95% CI `[0.0201, 0.0869]`, P(delta > 0) `1.000`.
- **ndcg_at_10:** mean delta `0.0263`, 95% CI `[0.0089, 0.0436]`, P(delta > 0) `0.999`.
- **map_at_100:** mean delta `0.0220`, 95% CI `[0.0101, 0.0354]`, P(delta > 0) `1.000`.

## Interpretation guidance

- Recall@K measures first-stage candidate coverage.
- MRR@10 rewards placing the first relevant result early.
- nDCG@10 evaluates the complete top-ranked ordering.
- MAP@100 summarizes precision across all relevant results.
- A positive reranking delta means the cross-encoder improved the ranking on average.
- Latency values depend on the local GPU, CPU, drivers and batch sizes.
