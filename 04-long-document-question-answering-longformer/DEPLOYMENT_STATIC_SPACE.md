# Hugging Face Static Space Deployment

## Why this folder is separate

The main project is a Python Longformer application. The `web/` folder is a
standalone browser deployment baseline designed for a free Hugging Face Static
Space. It uses a browser-compatible DistilBERT QA ONNX model and clearly labels
that architectural difference.

## Deploy

1. Open Hugging Face and create a new Space.
2. Select **Static HTML** as the SDK.
3. Suggested name: `long-document-qa-browser`.
4. Copy every file inside `web/` to the root of the Space repository.
5. Edit `src/config.js` and replace all username placeholders.
6. Commit the files.
7. The Space reads `web/README.md` metadata, runs:

```text
npm install --no-audit --no-fund && npm run build
```

8. The generated application is served from:

```text
dist/index.html
```

## Local validation

From the project folder:

```bash
cd web
npm install
npm test
npm run check
npm run build
npm run preview
```

## Files to copy to the Space

```text
README.md
index.html
package.json
vite.config.js
.gitignore
src/
public/
tests/
```

The `tests/` directory is not required at runtime but is useful for transparent
engineering and can remain in the Space repository.

## First-load behavior

The ONNX model downloads to the browser cache during the first session. The
first answer can therefore take longer. Later requests typically reuse cached
files. WebGPU is offered where supported, with an automatic WASM fallback.

## Security and privacy

The static application has no Python inference server. Document parsing and QA
run in the browser. However, model files are fetched from the Hugging Face Hub,
and public browser environments should still not be used with confidential or
regulated content.
