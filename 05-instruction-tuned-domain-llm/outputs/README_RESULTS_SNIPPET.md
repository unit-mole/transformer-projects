## Executed experiment results

The final run used **401** public-safe instruction examples with topic-grouped splits and evaluated **36** held-out prompts.

| Metric | Base FLAN-T5 | LoRA adapter |
|---|---:|---:|
| Held-out loss | 4.6528 | 3.9201 |
| Perplexity | 104.8759 | 50.4074 |
| BERTScore F1 | 0.6928 | 0.7404 |
| ROUGE-L | 0.0423 | 0.1438 |
| Semantic relevance | 0.2947 | 0.5923 |
| Instruction adherence | 0.4710 | 0.8452 |
| Automated hallucination-risk flag rate | 0.7500 | 0.1944 |
| Mean warm-cache latency, seconds | 0.2345 | 1.2847 |

Paired BERTScore F1 improvement: **0.0477**, with a 95% bootstrap interval of **[0.0235, 0.0741]**.

> BERTScore, ROUGE, embedding similarity, and heuristic risk flags do not prove factual correctness. See the per-example CSV files and complete the manual review template.