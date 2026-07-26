# Manual Relevance Analysis

Run the semantic search evaluation, export representative query-result pairs, and review each result using labels such as `highly_relevant`, `partially_relevant`, `not_relevant`, and `missed_relevant_result`.

Inspect:

- high-similarity false positives;
- semantically related but task-irrelevant passages;
- low-confidence correct results;
- effects of chunk boundaries and overlap;
- ambiguous queries;
- metadata or filtering errors;
- terminology that may require domain adaptation.

Do not publish numeric conclusions until the final corpus and actual model have been evaluated.
