# Hugging Face Static Space Deployment Guide

## Deployment architecture

Project 03 keeps the complete Python/Gradio implementation in GitHub and deploys
a separate Vite frontend as a free Hugging Face Static Space.

```text
GitHub project
├── Python implementation
├── Gradio local app
├── evaluation and tests
└── web/ Vite application
        ↓ npm run build
    web/dist/
        ↓ upload
Hugging Face Static Space
```

The Static Space performs real Transformer inference in the browser with:

```text
Xenova/all-MiniLM-L6-v2
Xenova/ms-marco-MiniLM-L-6-v2
Transformers.js
ONNX Runtime Web
```

## Create the Space

Use:

| Field | Value |
|---|---|
| Owner | `anmol-unitmole` |
| Space name | `cross-encoder-bi-encoder-ranking` |
| Short description | `Browser-based MiniLM retrieval and MS MARCO reranking with Transformers.js.` |
| License | MIT |
| SDK | Static |
| Template | Blank |
| Visibility | Public |

The project already contains a complete Vite application, so the Blank template
is appropriate.

## Space metadata

The metadata source is:

```text
web/public/README.md
```

Vite copies it to:

```text
web/dist/README.md
```

It contains:

```yaml
sdk: static
app_file: index.html
```

Do not use `sdk: gradio` for the live Space.

## Local browser testing

```bash
cd web
npm install
npm run check
npm test
npm run dev
```

Production verification:

```bash
npm run build
npm run preview
```

Then, from the project root:

```bash
python scripts/validate_dist.py
```

## Automatic GitHub deployment

Create the following in:

```text
GitHub repository
→ Settings
→ Secrets and variables
→ Actions
```

### Secret

```text
HF_TOKEN
```

Use a fine-grained token with write access to the target Space.

### Repository variable

```text
HF_SPACE_ID
```

Example:

```text
anmol-unitmole/cross-encoder-bi-encoder-ranking
```

The dedicated workflow:

```text
.github/workflows/03-cross-encoder-bi-encoder-ranking-system.yml
```

performs:

1. browser project validation;
2. JavaScript syntax checks;
3. browser metric tests;
4. Vite production build;
5. built-Space validation;
6. Python compilation;
7. Gradio import validation;
8. Python unit tests;
9. upload of `web/dist/` to the Static Space.

## Manual deployment

Build:

```bash
cd web
npm install
npm run build
cd ..
```

Set environment variables.

Windows PowerShell:

```powershell
$env:HF_TOKEN="<YOUR_TOKEN>"
$env:HF_SPACE_ID="anmol-unitmole/cross-encoder-bi-encoder-ranking"
```

macOS/Linux:

```bash
export HF_TOKEN="<YOUR_TOKEN>"
export HF_SPACE_ID="anmol-unitmole/cross-encoder-bi-encoder-ranking"
```

Deploy:

```bash
python -m pip install huggingface_hub
python scripts/deploy_static_space.py
```

The script creates or updates the target as a Static Space and uploads only the
contents of `web/dist/`.

## First-run model behavior

Each visitor's browser downloads quantized model assets from Hugging Face.

The first search includes:

- model download;
- tokenizer loading;
- ONNX initialization;
- document embedding;
- query embedding;
- retrieval;
- optional reranking.

Browser caching makes later searches faster.

## Troubleshooting

### Sync job is skipped

Verify that the GitHub repository variable exists:

```text
HF_SPACE_ID
```

### Authentication failure

Verify:

- `HF_TOKEN` is stored under Actions Secrets;
- the token has write permission;
- the target Space ID is correct;
- the token owner can write to the Space.

### Space opens but model loading fails

Possible causes:

- company network blocks model assets;
- WebAssembly is disabled;
- browser cache is corrupted;
- old browser version;
- interrupted first download.

Use a current Chrome, Edge, or Firefox browser and reload.

### Low-memory device

Use bi-encoder-only mode. This avoids loading the cross-encoder.

### Build does not include README metadata

Confirm this exists before building:

```text
web/public/README.md
```

Then delete `web/dist/` and rerun:

```bash
npm run build
```
