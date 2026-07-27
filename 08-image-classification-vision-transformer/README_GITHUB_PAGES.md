# GitHub Pages Deployment — Project 08

Project 08 is a fully static browser application. GitHub Pages publishes the files from:

```text
08-image-classification-vision-transformer/web/
```

The deployment workflow is:

```text
.github/workflows/08-image-classification-vision-transformer.yml
```

## Why the previous workflow failed

The project validation passed, but `actions/configure-pages` returned:

```text
Get Pages site failed: Not Found
```

That response means the repository did not yet have a GitHub Pages site. The normal workflow token cannot create the first Pages site. The corrected workflow therefore uses `enablement: true` with a separate fine-grained personal access token.

## Required one-time token setup

Create a fine-grained personal access token for only this repository:

1. Open GitHub **Settings** for your account.
2. Open **Developer settings → Personal access tokens → Fine-grained tokens**.
3. Create a token with repository access limited to `unit-mole/transformer-projects`.
4. Under repository permissions, set:
   - **Administration: Read and write**
   - **Pages: Read and write**
5. Copy the token.
6. Open `unit-mole/transformer-projects → Settings → Secrets and variables → Actions`.
7. Create a new repository secret named exactly:

```text
PAGES_DEPLOY_TOKEN
```

8. Paste the token as the secret value.

Do not place the token in a code file, commit, README, terminal screenshot, or Git history.

## What the corrected workflow does

The workflow now:

- validates the Project 08 Python and browser files;
- checks that `PAGES_DEPLOY_TOKEN` is available;
- creates/enables the repository Pages site when it does not yet exist;
- configures Pages for a GitHub Actions deployment;
- builds one combined static site artifact;
- preserves available browser demos for Projects 07, 08, and 09;
- uploads the Pages artifact;
- deploys through the official `github-pages` environment.

Project 08 will be published at:

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

Do not open `index.html` directly with a `file://` URL because browser module loading and model requests can be blocked.

## Troubleshooting

### Missing `PAGES_DEPLOY_TOKEN`

The workflow will now stop with a direct message telling you to add the repository secret. Create the fine-grained token and add it with the exact secret name.

### Token exists but enablement returns 403

Edit the fine-grained token and confirm that it is authorized for `unit-mole/transformer-projects` with both **Administration: Read and write** and **Pages: Read and write**.

### Validation passes but model inference fails in the browser

Open the browser developer console and inspect model CDN requests, WebAssembly support, WebGPU availability, and content-security restrictions.
