# Project 06 change log — confidence and evaluation upgrade

## Version 4.0.0

- Replaced the ambiguous `Confidence proxy: Not calibrated` presentation with
  `Answer confidence` and an explicit generative-model explanation.
- Added an optional generation confidence proxy derived from per-token model
  scores using a geometric-mean token-likelihood calculation.
- Kept the proxy explicitly uncalibrated and separate from factual accuracy.
- Added a deterministic 60-pair synthetic evaluation suite with 10 records per
  category across color, object, counting, yes/no, action or scene, and spatial
  questions.
- Added a browser Evaluation Lab with overall accuracy, category-wise accuracy,
  failure rate, average/minimum/maximum latency, JSON download, and failure review.
- Added evaluation generation and validation scripts plus CI tests.
- Updated model card, dataset card, Hugging Face guide, Space card, metadata,
  third-party notices, and project README.
- Updated Static Space asset versions to prevent stale browser caching.
