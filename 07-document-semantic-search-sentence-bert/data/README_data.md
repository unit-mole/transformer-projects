# Data Guide

Place only public, self-authored, redistributable documents in `raw_documents/`. Supported file types are `.md`, `.markdown`, and `.txt`.

Recommended sources:

- project README files;
- model and dataset cards;
- public deployment guides;
- public ML notes written by you;
- synthetic examples.

Never commit private company information, internal quality or GCS records, customer data, complaint investigations, CAPA records, SOPs with restricted access, personal information, credentials, or copyrighted text without permission.

Run:

```bash
python scripts/prepare_corpus.py
python scripts/generate_embeddings.py
python scripts/export_browser_data.py
```

The first command creates `corpus.json`, `document_chunks.json`, `metadata.json`, and corpus statistics. The second generates normalized Sentence-BERT vectors. The third copies browser-safe artifacts to `web/data/`.
