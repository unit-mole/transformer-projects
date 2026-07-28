# Data documentation

This repository contains only safe project-generated images.

## Interactive smoke-test data

- 3 synthetic images
- 3 VQA-style question-answer records
- categories: color, number, and object

## Browser evaluation data

- 60 synthetic image-question pairs
- 10 records each for color, object, counting, yes/no, action or scene, and spatial relationships
- JSON and CSV formats
- deterministic image-generation script
- no real people, personal information, or private content

Run:

```bash
python scripts/generate_synthetic_evaluation_set.py --check
```

to validate the committed dataset.

The full VQA v2 dataset is intentionally excluded because it is large and has
its own usage terms. The Python utilities remain compatible with VQA v2-style
fields when a separately obtained dataset is used.

Do not add personal photos, IDs, medical images, confidential workplace images,
copyrighted images without permission, or images containing sensitive personal
data.
