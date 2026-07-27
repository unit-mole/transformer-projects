# GitHub Pages Deployment — Project 08

This project is a static browser application. GitHub Pages publishes the files from:

```text
08-image-classification-vision-transformer/web/
```

The deployment workflow is:

```text
.github/workflows/08-image-classification-vision-transformer.yml
```

## Required one-time repository setting

Before the first successful deployment, enable GitHub Pages for the repository:

1. Open `https://github.com/unit-mole/transformer-projects`.
2. Select **Settings**.
3. In the left sidebar, select **Pages**.
4. Under **Build and deployment**, set **Source** to **GitHub Actions**.
5. Return to **Actions** and re-run the failed workflow, or push the corrected workflow.

The `Get Pages site failed: Not Found` message occurs when the repository has not yet been enabled for GitHub Pages. This is a repository setting, not a Python, JavaScript, or model error.

## Deployment behavior

The corrected workflow:

- validates Project 08;
- builds one combined GitHub Pages artifact;
- publishes Projects 07, 08, and 09 under separate subpaths when their `web/index.html` files exist;
- preserves the other GitHub Pages projects instead of replacing the whole site with only Project 08;
- uses current Node.js 24-compatible action versions where available;
- deploys Project 08 at:

```text
https://unit-mole.github.io/transformer-projects/08-image-classification-vision-transformer/
```

## Local browser test

From the repository root:

```cmd
cd 08-image-classification-vision-transformer\web
python -m http.server 8000
```

Open:

```text
http://localhost:8000
```

Do not open `index.html` directly with a `file://` URL because browser module loading and model requests may be blocked.

## Troubleshooting

### Configure Pages fails with `Not Found`

Set **Settings → Pages → Source → GitHub Actions**, then re-run the workflow.

### Validation passes but deployment fails

The application files are valid. Check the repository Pages source and the workflow permissions:

```yaml
permissions:
  contents: read
  pages: write
  id-token: write
```

### Project 07 or Project 09 disappears

GitHub Pages deploys one artifact for the entire repository. Use the corrected combined-site workflow, which copies all available GitHub Pages projects into the same `_site` artifact.

### The deployed page loads but inference does not

Open the browser developer console and check blocked network requests, model CDN access, WebAssembly support, and content-security restrictions.
