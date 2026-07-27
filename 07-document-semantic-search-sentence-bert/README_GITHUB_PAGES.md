# GitHub Pages Deployment Guide

This project uses a **branch-based GitHub Pages deployment**. The workflow validates Project 07, assembles the static site, and force-publishes the generated site to the repository's `gh-pages` branch.

This method intentionally does **not** use:

- `actions/configure-pages`
- `actions/upload-pages-artifact`
- `actions/deploy-pages`
- the GitHub Pages REST API
- a Personal Access Token

Therefore, the workflow is not blocked by the `Get Pages site failed: Not Found` error that occurs when a Pages site has not yet been enabled.

## Files deployed

The workflow copies:

```text
07-document-semantic-search-sentence-bert/web/
```

into this published route:

```text
/07-document-semantic-search-sentence-bert/
```

It also creates a root `index.html` that redirects visitors to Project 07 and adds `.nojekyll`.

## First deployment

1. Push Project 07 and `.github/workflows/07-document-semantic-search-sentence-bert.yml` to `main`.
2. Open the repository's **Actions** tab.
3. Confirm that **Test and validate Project 07** succeeds.
4. Confirm that **Publish Project 07 to gh-pages branch** succeeds.
5. Open **Settings → Pages**.
6. Under **Build and deployment**, choose **Deploy from a branch**.
7. Select branch **`gh-pages`** and folder **`/ (root)`**.
8. Click **Save**.

The one-time Pages setting is a repository configuration; it cannot be reliably created by a normal workflow using the default `GITHUB_TOKEN`.

## Live URL

After GitHub finishes publishing, the project URL is expected to be:

```text
https://unit-mole.github.io/transformer-projects/07-document-semantic-search-sentence-bert/
```

## Local testing

```bash
cd 07-document-semantic-search-sentence-bert/web
python -m http.server 8000
```

Open:

```text
http://localhost:8000
```

Do not open `index.html` directly with `file://`, because browsers may block JavaScript module and JSON requests.

## Troubleshooting

### The workflow succeeds, but the site is not available

Open **Settings → Pages** and verify:

```text
Source: Deploy from a branch
Branch: gh-pages
Folder: / (root)
```

### The workflow cannot push to `gh-pages`

Open **Settings → Actions → General → Workflow permissions** and make sure workflows are allowed to use write permissions. The workflow explicitly requests `contents: write`.

### JSON files fail to load

Verify that these files exist on the `gh-pages` branch:

```text
07-document-semantic-search-sentence-bert/data/document_chunks.json
07-document-semantic-search-sentence-bert/data/embeddings.json
07-document-semantic-search-sentence-bert/data/metadata.json
```

The browser app uses relative paths, so it works from the nested GitHub Pages route.

### Browser embedding model does not load

The app attempts to load a browser-compatible Sentence Transformer through Transformers.js. Network restrictions, browser privacy settings, or CDN availability can block the model. The interface displays the active search mode and can use its documented fallback behavior.
