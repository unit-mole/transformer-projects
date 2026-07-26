# Generated outputs

The committed files in this directory are honest placeholders. They do not
contain invented performance values.

Run:

```bash
python scripts/preprocess_documents.py
python scripts/evaluate_model.py
python scripts/run_context_analysis.py
```

The scripts generate:

- `preprocessed_sample_qa.csv`
- `qa_examples.csv`
- `model_metrics.json`
- `manual_error_analysis.md`
- `context_length_analysis.csv`
- `context_length_analysis.json`
- `context_length_analysis.png`

Publish metric values in the README only after these scripts run successfully
on the selected checkpoint and dataset.
