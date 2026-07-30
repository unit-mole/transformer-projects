# Manual Error Analysis

Complete this file after the full evaluation run. Select examples from `outputs/rag_answer_examples.csv` and the claim-level JSON records.

| Question ID | Error type | Observed behavior | Supporting or conflicting source | Root cause | Corrective action | Retest result |
|---|---|---|---|---|---|---|
| `<id>` | Weak retrieval / missing citation / unsupported claim / wrong project / refusal error | `<observation>` | `<chunk IDs>` | `<data, chunking, retriever, reranker, prompt, threshold>` | `<change>` | `<pass/fail>` |

Required review groups:

1. Strong grounded answers.
2. Multi-project questions.
3. Paraphrased questions.
4. Similar-project confusion.
5. Missing or incorrect citations.
6. Unsupported/private-information questions.
7. Stale or incomplete documentation.
8. High-latency requests.
