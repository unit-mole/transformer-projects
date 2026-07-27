# GitHub Pages deployment guide — `main /docs`

Project 09 follows the permanent Pages structure already configured for the `transformer-projects` repository.

```text
Source: Deploy from a branch
Branch: main
Folder: /docs
```

Do not change these repository settings for Project 09.

## Final deployment layout

```text
transformer-projects/
├── 09-vision-language-image-text-retrieval-clip/
│   ├── web/                                      # development source
│   ├── src/
│   ├── scripts/
│   ├── tests/
│   └── README.md
├── docs/
│   ├── .nojekyll
│   └── 09-vision-language-image-text-retrieval-clip/  # published copy
│       ├── index.html
│       ├── style.css
│       ├── app.js
│       ├── clip_preprocessing.js
│       ├── clip_inference.js
│       ├── retrieval.js
│       ├── zero_shot.js
│       ├── metadata.json
│       ├── zero_shot_labels.json
│       ├── data/
│       ├── sample_images/
│       └── model/
└── .github/workflows/
    └── 09-vision-language-image-text-retrieval-clip.yml
```

The deployed URL is:

```text
https://unit-mole.github.io/transformer-projects/09-vision-language-image-text-retrieval-clip/
```

## Development and publishing rule

Always edit the application inside:

```text
09-vision-language-image-text-retrieval-clip/web/
```

Do not manually maintain two different versions. After each browser-app change, synchronize the deployment copy:

```bash
cd 09-vision-language-image-text-retrieval-clip
python scripts/sync_docs.py
```

The command replaces:

```text
docs/09-vision-language-image-text-retrieval-clip/
```

with an exact copy of `web/` and preserves `docs/.nojekyll`.

Confirm that both copies are identical:

```bash
python scripts/sync_docs.py --check
```

## Relative-path requirement

Because Project 09 is hosted below a repository subfolder, local static assets must use relative paths:

```html
<link rel="stylesheet" href="./style.css">
<script type="module" src="./app.js"></script>
```

```javascript
fetch('./metadata.json');
fetch('./data/image_gallery.json');
```

```json
{
  "image_path": "./sample_images/red_car.png"
}
```

Do not use repository-root paths such as `/style.css`, `/model/model.onnx`, or `/sample_images/red_car.png`. The included `scripts/check_relative_paths.py` and CI workflow reject those references.

## Prepare the gallery

```bash
python scripts/prepare_gallery.py
```

This validates the sample images, captions, metadata, required public-demo fields, and image references.

## Generate CLIP gallery embeddings

Install the optional model dependencies:

```bash
pip install -r requirements-model.txt
python scripts/generate_image_embeddings.py
```

The generated normalized vectors are written to:

```text
web/data/image_embeddings.json
```

Then synchronize the deployment folder:

```bash
python scripts/sync_docs.py
```

When the vectors list is intentionally empty, the app can generate the small gallery embeddings in the browser and cache them locally. The interface labels its caption-search fallback clearly when the CLIP runtime cannot load.

## Browser model approach

The default app loads the quantized `Xenova/clip-vit-base-patch32` model through Transformers.js. This provides browser-compatible ONNX text and vision encoders without a Python backend or server-side API.

The `web/model/` directory contains a manifest and self-hosting instructions. Large ONNX files are not bundled in this repository package because they can exceed practical Git and Pages limits. When custom browser-compatible model files are added, keep every model reference relative and copy the same model folder into the `docs` deployment copy through `sync_docs.py`.

## Test the development app locally

From the Project 09 folder:

```bash
python scripts/run_local_web_server.py
```

Open:

```text
http://localhost:8000
```

You may also use:

```bash
npm install
npm run serve
```

Do not open `index.html` directly through `file://`, because browsers can block JavaScript module, JSON, WASM, and model requests.

## Test the exact deployment copy locally

First synchronize it:

```bash
python scripts/sync_docs.py
```

Then run:

```bash
npm run serve:docs
```

Open `http://localhost:8000` and verify that the exact `docs/09-.../` files work before committing.

## Run all validation

```bash
pip install -r requirements.txt
pytest -q
python scripts/prepare_gallery.py
python scripts/export_browser_assets.py
python scripts/sync_docs.py --check
python scripts/check_relative_paths.py
node scripts/validate_web_assets.mjs --target web
node scripts/validate_web_assets.mjs --target docs
```

Or, after `npm install`:

```bash
npm run validate
```

## Git workflow

From the repository root:

```bash
git add 09-vision-language-image-text-retrieval-clip
git add docs/09-vision-language-image-text-retrieval-clip
git add docs/.nojekyll
git add .github/workflows/09-vision-language-image-text-retrieval-clip.yml
git commit -m "Add Project 09 CLIP GitHub Pages application"
git push origin main
```

GitHub automatically rebuilds the site because the configured `/docs` publishing folder changed.

## Validation-only GitHub Actions

The Project 09 workflow is intentionally limited to:

- Python tests
- source data validation
- browser asset validation
- `web/` and `docs/09-.../` synchronization checks
- relative-path checks

It does not use:

```text
actions/configure-pages
actions/deploy-pages
PAGES_DEPLOY_TOKEN
gh-pages branch deployment
```

The repository's built-in Pages workflow remains responsible for publishing `main /docs`.

## Final verification

After the push completes, open:

```text
https://unit-mole.github.io/transformer-projects/09-vision-language-image-text-retrieval-clip/
```

Verify:

1. The page loads without a 404.
2. CSS and JavaScript load from the Project 09 subfolder.
3. Gallery images and JSON assets load.
4. Sample-query retrieval works.
5. The model-status card reports CLIP loading or a clearly labeled fallback.
6. Uploaded-image validation works.
7. Zero-shot candidate labels can be edited.
8. Browser developer tools show no repository-root path errors.

## Troubleshooting

### The Project 09 URL returns 404

Confirm that `docs/09-vision-language-image-text-retrieval-clip/index.html` exists on the `main` branch and that the repository remains configured for `main /docs`.

### CSS, JavaScript, JSON, model, or image requests return 404

Search for paths beginning with a single `/`. Replace them with `./` paths and run:

```bash
python scripts/check_relative_paths.py
```

### The deployment shows an older version

Run `python scripts/sync_docs.py`, commit the changed `docs/09-.../` files, push to `main`, and wait for the repository's built-in Pages build to finish.

### CLIP cannot load

Check internet access, browser memory, the browser console, remote model availability, and WASM support. The static gallery and labeled caption baseline remain available even when the browser model cannot initialize.

### JSON fails during local testing

Use the included local HTTP server rather than opening the HTML file directly.
