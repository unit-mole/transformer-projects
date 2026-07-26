# Manual Relevance Analysis

Complete this file after running `python scripts/evaluate_model.py`.

| Query | Strong retrieval | Weak retrieval | Reranking change | Human assessment |
|---|---|---|---|---|
| `<query>` | `<relevant result>` | `<incorrect result>` | `<rank movement>` | `<observation>` |

## Review checklist

- Ambiguous query
- Keyword match but wrong meaning
- Semantic match but wrong domain
- Relevant document absent from candidate set
- Cross-encoder improvement
- Cross-encoder regression
- Cross-encoder overconfidence
- Missing context or overly broad query
- Sensitive or unfair job-ranking behavior

Do not overstate capability. Record failures alongside successful examples.
