# GitHub Pages Deployment — Project 08

## Deployment method used

This project is published from the repository's `main` branch and `/docs` folder. It does **not** use `actions/configure-pages`, `actions/deploy-pages`, a personal access token, or an Actions secret.

The browser application remains in:

```text
08-image-classification-vision-transformer/web/
```

The deployable copy is stored in:

```text
docs/08-image-classification-vision-transformer/
```

## Required one-time GitHub setting

After pushing the files, open the repository and configure:

```text
Settings
→ Pages
→ Build and deployment
→ Source: Deploy from a branch
→ Branch: main
→ Folder: /docs
→ Save
```

This setting creates the GitHub Pages site. It cannot be created by ordinary repository files alone.

## Live URL

```text
https://unit-mole.github.io/transformer-projects/08-image-classification-vision-transformer/
```

## Updating the browser application

After changing files in `08-image-classification-vision-transformer/web/`, copy the same files into `docs/08-image-classification-vision-transformer/` before committing. The validation workflow checks that both folders match.

Windows CMD example from the repository root:

```cmd
rmdir /s /q "docs\08-image-classification-vision-transformer"
mkdir "docs\08-image-classification-vision-transformer"
xcopy /e /i /y "08-image-classification-vision-transformer\web\*" "docs\08-image-classification-vision-transformer\"
```

## Local test

From the repository root:

```cmd
python -m http.server 8000 --directory docs
```

Open:

```text
http://localhost:8000/08-image-classification-vision-transformer/
```

## Why the previous workflows failed

The previous workflows called `actions/configure-pages` before the repository had a Pages site. GitHub returned `404 Not Found`. A later workflow required a custom `PAGES_DEPLOY_TOKEN`, so it intentionally stopped when that secret was missing. This branch-based `/docs` deployment removes both dependencies.
