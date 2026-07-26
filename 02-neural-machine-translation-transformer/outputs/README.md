# Outputs

Committed JSON files are honest placeholders with `null` values. Run:

```bash
python scripts/evaluate_model.py \
  --input data/sample_translation_pairs.csv \
  --output-dir outputs/generated
```

Generated files are excluded from Git by default until reviewed.
