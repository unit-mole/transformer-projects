# Portfolio Corpus Safety and Preparation

This directory is for **public portfolio documentation only**.

## Safe sources

- Public GitHub project READMEs.
- Public model cards, dataset cards, deployment guides, and evaluation summaries.
- Verified public model, dataset, metric, and deployment descriptions.

## Prohibited sources

- Hach, GCS, quality-case, customer, supplier, CAPA, or internal corrective-action files.
- Private email, credentials, API keys, personal identifiers, unpublished work documents, proprietary manuals, or confidential business data.
- Documents whose redistribution rights are unclear.

## Expected layout

The GitHub collector writes:

```text
data/raw_portfolio_docs/<category>/<repository>/<original path>
```

This preserves repository and source-path traceability. Review collected files before preprocessing.

## Rebuild order

```bash
python scripts/collect_github_portfolio_docs.py --clean
python scripts/prepare_corpus.py
python scripts/generate_embeddings.py --provider minilm --device cuda
python scripts/export_vector_store.py
python scripts/run_retrieval_benchmark.py --device cuda --include-e5 --include-reranker
python scripts/evaluate_answers.py --device cuda --generator flan-t5-base
python scripts/build_evaluation_summary.py
python scripts/create_evaluation_charts.py
```

Every corpus modification invalidates the previous embedding and evaluation results. Regenerate all downstream artifacts.
