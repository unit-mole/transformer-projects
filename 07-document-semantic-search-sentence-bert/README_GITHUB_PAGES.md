# GitHub Pages Deployment Guide

This project is a static HTML/CSS/JavaScript application. It does not require a Python backend, API server, vector database, Streamlit, Gradio, or paid hosting.

## Final route

```text
https://unit-mole.github.io/transformer-projects/07-document-semantic-search-sentence-bert/
```

The repository root URL redirects to the Project 07 route:

```text
https://unit-mole.github.io/transformer-projects/
```

## Important one-time GitHub setting

GitHub Pages is a repository-level service and must exist before `actions/configure-pages` can read it. A workflow file cannot create the Pages site with the default `GITHUB_TOKEN`.

Choose either method below once:

### Method A — GitHub interface

1. Open the `unit-mole/transformer-projects` repository.
2. Select **Settings → Pages**.
3. Under **Build and deployment**, set **Source** to **GitHub Actions**.
4. Save the setting.

### Method B — GitHub CLI

After `gh auth login`, run:

```cmd
gh api --method POST repos/unit-mole/transformer-projects/pages -f build_type=workflow 2>nul || gh api --method PUT repos/unit-mole/transformer-projects/pages -f build_type=workflow
```

A `Get Pages site failed: Not Found` error means the Pages site still has not been enabled. Replacing the YAML alone cannot correct that repository-level 404.

## Files that must be pushed

Push only these paths:

```text
07-document-semantic-search-sentence-bert/
.github/workflows/07-document-semantic-search-sentence-bert.yml
```

The corrected workflow does not depend on a root-level `pages/` folder. It creates a temporary `_site` directory during the Actions run, copies only Project 07's `web/` application into the nested public route, adds a repository-root redirect, uploads the Pages artifact, and deploys it.

## Before deployment

1. Keep only public, redistributable documents in `data/raw_documents/`.
2. Run `python scripts/prepare_corpus.py` when the corpus changes.
3. Run `python scripts/generate_embeddings.py` for precomputed production embeddings.
4. Run `python scripts/export_browser_data.py`.
5. Confirm that these files exist:
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
6. Test locally from the `web/` directory with `python -m http.server 8000`.

## Deployment steps

1. Enable **Settings → Pages → Source → GitHub Actions**.
2. Replace the existing Project 07 workflow with the corrected YAML.
3. Push Project 07 and that workflow to `main`.
4. Open **Actions → 07 Document Semantic Search Sentence BERT CI and Pages**.
5. Confirm that `Test and validate Project 07` succeeds.
6. Confirm that `Deploy Project 07 to GitHub Pages` succeeds.
7. Open the deployment URL shown in the workflow summary.

## Troubleshooting

### `Get Pages site failed: Not Found`

Enable GitHub Pages under **Settings → Pages** and choose **GitHub Actions** as the source. Then rerun the failed job.

### JSON files return 404

- Verify that JSON files are committed under `web/data/`.
- Keep browser paths relative, such as `./data/document_chunks.json`.
- Check filename capitalization because GitHub Pages paths are case-sensitive.
- Inspect the browser Network tab for the exact failed URL.

### The browser model does not load

- Confirm that the browser permits requests to the Hugging Face Hub and jsDelivr CDN.
- Temporarily disable restrictive content blockers for the demo.
- Test with a current Chrome, Edge, or Firefox release.
- The app switches visibly to keyword fallback when semantic model initialization fails.

### First load is slow

The first visit may download a quantized ONNX model and generate document embeddings. Later visits can reuse the browser cache and locally cached vectors. Precomputed embeddings provide a faster public demo.

## README link

```markdown
[Open the live GitHub Pages demo](https://unit-mole.github.io/transformer-projects/07-document-semantic-search-sentence-bert/)
```
