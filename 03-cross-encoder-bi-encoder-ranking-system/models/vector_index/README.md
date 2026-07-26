# Vector Index

Run `python scripts/build_index.py` to generate:

- `embeddings.npy`
- `document_ids.json`
- `index_metadata.json`

The public demo uses a small NumPy cosine-similarity index to avoid FAISS
installation problems on lightweight CPU deployment. The application builds this
small sample index automatically only when a compatible saved index is absent.

Generated index files are ignored by Git by default. Commit them only when their
size is appropriate and their source data is safe to redistribute.
