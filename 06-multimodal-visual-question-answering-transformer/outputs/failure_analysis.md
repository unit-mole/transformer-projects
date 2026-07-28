# Failure analysis

Status: **not evaluated**

Run the 60-question browser Evaluation Lab and download its JSON report. Review
incorrect or failed records by category, especially:

- wrong color;
- wrong object;
- wrong count;
- wrong yes/no answer;
- action or scene misunderstanding;
- spatial-relation error;
- empty or failed generation;
- unusually high latency;
- high token-likelihood proxy paired with an incorrect answer.

The final item is particularly important: it demonstrates why a generation
confidence proxy must not be treated as a calibrated correctness probability.
