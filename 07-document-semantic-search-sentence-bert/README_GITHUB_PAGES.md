# GitHub Pages Deployment Guide

This project is intentionally built as a static HTML/CSS/JavaScript application. It does not require a Python backend, API server, vector database, or paid hosting.

## Final route

```text
https://unit-mole.github.io/transformer-projects/07-document-semantic-search-sentence-bert/
```

## Before deployment

1. Keep only public, redistributable documents in `data/raw_documents/`.
2. Run `python scripts/prepare_corpus.py`.
3. For the production path, run `python scripts/generate_embeddings.py`.
4. Run `python scripts/export_browser_data.py`.
5. Confirm these files exist:
   - `web/index.html`
   - `web/style.css`
   - `web/app.js`
   - `web/search.js`
   - `web/embeddings.js`
   - `web/data/corpus.json`
   - `web/data/document_chunks.json`
   - `web/data/embeddings.json`
   - `web/data/evaluation_queries.json`
   - `web/data/metadata.json`
6. Test with `python -m http.server 8000` from the `web/` directory.

## Publish with the included workflow

1. Push the project and `.github/workflows/07-document-semantic-search-sentence-bert.yml` to the `main` branch.
2. Open the repository on GitHub.
3. Select **Settings → Pages**.
4. Under **Build and deployment**, choose **GitHub Actions** as the source.
5. Open **Actions** and run the workflow manually, or push a change inside this project.
6. Wait for both `quality` and `deploy` jobs to finish.
7. Open the deployment URL from the workflow summary.

The workflow creates `_site`, copies the root demo hub from `pages/`, and copies every matching `NN-project-name/web/` directory to a separate public subpath. This avoids hard-coded absolute asset paths and supports future GitHub Pages projects 08 and 09 in the same repository.

## Why not deploy the nested folder directly from a branch?

GitHub Pages branch deployment normally publishes from the repository root or `/docs`. This monorepo contains several projects, so a custom Actions deployment provides a cleaner route for each static demo while keeping source files in their project folders.

## Troubleshooting

### JSON files return 404

- Verify the JSON files are committed under `web/data/`.
- Use relative paths such as `./data/document_chunks.json`; do not begin paths with `/`.
- Check filename capitalization because GitHub Pages is case-sensitive.
- Inspect the browser Network tab for the exact failing URL.

### The model does not load

- Confirm the browser allows requests to the Hugging Face Hub and jsDelivr CDN.
- Disable restrictive content blockers for the demo temporarily.
- Test on a current Chrome, Edge, or Firefox release.
- The app will clearly switch to keyword fallback if semantic model initialization fails.

### First load is slow

The first visit downloads a quantized ONNX model and may generate document embeddings. Subsequent visits can reuse the browser cache and locally cached vectors. For a faster public demo, precompute document embeddings using the Python script and commit the exported browser payload.

### The workflow passes but Pages is unavailable

Open **Settings → Pages** and confirm the source is **GitHub Actions**. Also check that Pages is permitted for the repository and that the deployment job has `pages: write` and `id-token: write` permissions.

## Add the link to the README

Use:

```markdown
[Open the live GitHub Pages demo](https://unit-mole.github.io/transformer-projects/07-document-semantic-search-sentence-bert/)
```
