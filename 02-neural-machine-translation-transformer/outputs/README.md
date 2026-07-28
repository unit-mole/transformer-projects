# Outputs

Committed JSON files are honest placeholders with `null` values. Run:

```bash
python scripts/evaluate_model.py \
  --input data/sample_translation_pairs.csv \
  --output-dir outputs/generated
```

Generated files are excluded from Git by default until reviewed.

## Portfolio-grade evaluation

Run `notebooks/03_portfolio_grade_marianmt_finetuning_evaluation.ipynb` to create real pretrained-versus-fine-tuned metrics under `outputs/portfolio_evaluation/` and populate the root JSON placeholders. Do not manually type metric values.
