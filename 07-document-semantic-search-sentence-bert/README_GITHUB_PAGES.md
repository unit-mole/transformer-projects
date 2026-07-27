# GitHub Pages Deployment Guide — `main` / `docs`

This repository uses one permanent GitHub Pages configuration for all static browser projects:

```text
Source: Deploy from a branch
Branch: main
Folder: /docs
```

GitHub Pages publishes the content already committed under `/docs`. Project-specific workflows validate files and tests only; they do not deploy the site.

## Required Project 07 structure

```text
transformer-projects/
├── 07-document-semantic-search-sentence-bert/
│   └── web/
│       ├── index.html
│       ├── style.css
│       ├── app.js
│       ├── search.js
│       ├── embeddings.js
│       ├── metadata.json
│       └── data/
│           ├── corpus.json
│           ├── document_chunks.json
│           ├── embeddings.json
│           ├── evaluation_queries.json
│           └── metadata.json
├── docs/
│   ├── .nojekyll
│   ├── index.html
│   └── 07-document-semantic-search-sentence-bert/
│       └── exact copy of the files in Project 07 web/
└── .github/workflows/
    └── 07-document-semantic-search-sentence-bert.yml
```

## Development and deployment copies

The editable application remains here:

```text
07-document-semantic-search-sentence-bert/web/
```

The GitHub Pages copy is here:

```text
docs/07-document-semantic-search-sentence-bert/
```

Do not manually edit the `/docs` copy. Edit `web/` and synchronize it using:

```bash
python 07-document-semantic-search-sentence-bert/scripts/sync_docs_site.py
```

The script replaces the Project 07 deployment directory with an exact copy of `web/` and ensures that `docs/.nojekyll` exists.

Verify that both copies are identical:

```bash
python 07-document-semantic-search-sentence-bert/scripts/sync_docs_site.py --check
```

## Complete publishing sequence

From the repository root:

```bash
python 07-document-semantic-search-sentence-bert/scripts/sync_docs_site.py
pytest 07-document-semantic-search-sentence-bert/tests -q
node --check 07-document-semantic-search-sentence-bert/web/app.js
node --check 07-document-semantic-search-sentence-bert/web/search.js
node --check 07-document-semantic-search-sentence-bert/web/embeddings.js
```

Then stage Project 07, its deployment mirror, and its validation workflow:

```bash
git add -A -- \
  "07-document-semantic-search-sentence-bert" \
  "docs/07-document-semantic-search-sentence-bert" \
  "docs/.nojekyll" \
  ".github/workflows/07-document-semantic-search-sentence-bert.yml"
```

Commit and push to `main`. Because `/docs` changed on the configured publishing branch, GitHub runs its built-in `pages build and deployment` workflow.

## Live URL

```text
https://unit-mole.github.io/transformer-projects/07-document-semantic-search-sentence-bert/
```

## Relative-path rule

All browser files must use relative asset and data references:

```javascript
fetch("./data/document_chunks.json")
import { rankSemantic } from "./search.js"
```

```html
<link rel="stylesheet" href="./style.css">
<script type="module" src="./app.js"></script>
```

Do not use repository-root paths such as:

```text
/data/document_chunks.json
/model/model.json
```

A leading slash points to the root of `unit-mole.github.io`, not to the Project 07 subdirectory.

## Workflow responsibility

`.github/workflows/07-document-semantic-search-sentence-bert.yml` is intentionally validation-only. It:

- runs only when Project 07, its `/docs` mirror, or its own workflow changes;
- runs Python tests;
- checks JavaScript syntax;
- validates required JSON and static files;
- verifies that `web/` and the `/docs` Project 07 copy are identical;
- never modifies Pages settings or publishes another branch.

It must not contain:

```text
actions/configure-pages
actions/deploy-pages
actions/upload-pages-artifact
PAGES_DEPLOY_TOKEN
gh-pages branch publishing
```

## Troubleshooting

### The project URL returns 404

Check all four conditions:

1. Repository Settings → Pages shows `main` and `/docs`.
2. `docs/07-document-semantic-search-sentence-bert/index.html` exists on the remote `main` branch.
3. The latest built-in `pages build and deployment` run is green.
4. The URL exactly matches the folder name, including hyphens and the trailing slash.

### Project 07 CI passes but the website does not update

CI only validates. Confirm that the `/docs` copy was staged and committed. Run:

```bash
python 07-document-semantic-search-sentence-bert/scripts/sync_docs_site.py --check
git status
```

### Project 08 runs when only Project 07 changes

The Project 08 workflow must include `paths` filters for only Project 08 and its own workflow file. A broad `on: push` trigger runs on every push to `main`.

### JSON fails to load

Serve locally with HTTP rather than opening the HTML file directly:

```bash
cd docs
python -m http.server 8000
```

Open:

```text
http://localhost:8000/07-document-semantic-search-sentence-bert/
```

### The model does not load

The application automatically labels and uses its keyword fallback when the Transformers.js CDN or browser model cannot load. Check browser developer tools for network, CSP, WebAssembly, or storage errors.
