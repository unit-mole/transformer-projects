# Portfolio-grade evaluation dataset

The evaluation workflow loads `cfilt/iitb-english-hindi` directly from the Hugging Face Hub. The repository does not redistribute the full corpus.

Default `portfolio` profile:

- 50,000 training pairs sampled from the official training split
- all 520 validation pairs
- 1,000 held-out test pairs sampled from the official test split
- deterministic seed: 42

The IIT Bombay English-Hindi Parallel Corpus is licensed for non-commercial use. Review the dataset card and original CFILT license before any use beyond an educational portfolio.

Generated split CSV files are written to `outputs/portfolio_evaluation/datasets/` so every metric can be reproduced from the exact pairs used during the run.
