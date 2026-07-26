# Dataset notes

## Included files

- `sample_translation_pairs.csv`: small, hand-authored English–Hindi examples for smoke tests and demonstrations.
- `sample_batch_translation.csv`: mixed English, Hindi, and mixed-script inputs for the batch interface.

These samples are not a claim of model training or benchmark coverage.

## Full dataset support

Recommended public corpus reference:

```text
cfilt/iitb-english-hindi
```

Expected feature:

```python
{"translation": {"en": "...", "hi": "..."}}
```

The repository does not redistribute the full dataset. Review the dataset card and source terms before downloading, training, or sharing derived artifacts.

## Custom CSV

Accepted common column names include:

```text
english / hindi
en / hi
sentence_en / sentence_hi
source / target
source_text / target_text
```

For ambiguous `source` and `target` files, pass the intended direction explicitly.

## Safety

Do not upload customer complaints, service cases, company reports, personal data, copyrighted text, or confidential records to a public repository or public Space.
