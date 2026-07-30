# Vercel Deployment Guide — Project 10

The production app is a Next.js App Router application. All expensive corpus work—document collection, cleaning, chunking, Transformer embedding generation, and formal evaluation—runs offline on the local RTX GPU. Vercel loads committed static JSON artifacts and uses server-side API routes for question embedding, retrieval, and optional hosted generation.

## Deployment prerequisites

Do not deploy the final portfolio version until:

1. `notebooks/01-build-and-evaluate-transformer-rag.ipynb` has run successfully.
2. `public/data/metadata.json` reports a real Transformer embedding model.
3. All chunk and embedding counts match.
4. `public/data/evaluation_summary.json` contains measured—not invented—results.
5. `npm run validate:data`, `npm test`, `npm run typecheck`, and `npm run build` pass.

A preview deployment with starter hash vectors is acceptable for UI testing, but it must remain labeled as a starter build.

## 1. Generate final static artifacts

Use the GPU notebook or:

```powershell
python scripts/collect_github_portfolio_docs.py --clean
python scripts/run_full_evaluation.py `
  --device cuda `
  --generator flan-t5-base `
  --include-e5 `
  --include-reranker
```

The following runtime files must exist and be committed:

```text
public/data/document_chunks.json
public/data/embeddings.json
public/data/metadata.json
public/data/evaluation_questions.json
public/data/evaluation_summary.json
```

No Python process or notebook is required after deployment.

## 2. Confirm model consistency

Open `public/data/metadata.json` and verify that the document-embedding model matches the model that Vercel will use for question embeddings.

Example:

```json
{
  "embedding": {
    "provider": "huggingface-feature-extraction",
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "dimension": 384,
    "normalized": true,
    "queryPrefix": "",
    "passagePrefix": ""
  }
}
```

For E5, the `queryPrefix` and `passagePrefix` fields must be preserved. Never mix MiniLM document vectors with E5 query vectors.

## 3. Test the production build locally

```powershell
npm install
copy .env.example .env.local
npm run validate:data
npm test
npm run typecheck
npm run build
npm run dev
```

Verify:

- `http://localhost:3000/api/health`
- `http://localhost:3000/api/evaluation`
- `POST /api/retrieve`
- `POST /api/chat`
- source cards and `[S#]` citations
- weak-evidence refusal
- category/deployment filters
- model and corpus readiness labels

## 4. Environment variables

### Required for real Transformer query embeddings

```text
HF_API_TOKEN=your_server_side_token
HF_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### Optional hosted instruction generation

```text
USE_HF_GENERATOR=true
HF_GENERATOR_MODEL=your-supported-instruction-model
```

### Optional confidence control

```text
MIN_RETRIEVAL_SCORE=0.15
```

Never prefix secrets with `NEXT_PUBLIC_`.

## 5. Import the monorepo into Vercel

1. Push all final files to `unit-mole/transformer-projects`.
2. In Vercel, select **Add New → Project**.
3. Import the GitHub repository.
4. Set **Root Directory** to `10-ai-portfolio-rag-assistant`.
5. Keep **Framework Preset: Next.js**.
6. Use **Install Command: `npm install`**.
7. Use **Build Command: `npm run build`**.
8. Add environment variables to Preview and Production.
9. Deploy.

The included `vercel.json` assigns explicit maximum durations to the chat, retrieval, and health routes. Hosted generation may take longer than local extractive generation, so test the selected provider before publishing.

## 6. Post-deployment checks

Open:

```text
https://YOUR-PROJECT.vercel.app/api/health
```

Expected indicators:

```json
{
  "status": "ok",
  "transformerReady": true,
  "embeddingProvider": "huggingface-feature-extraction"
}
```

Then test at least:

1. A direct project question.
2. A paraphrased question.
3. A multi-project comparison.
4. A deployment question.
5. An unsupported/private-information question.

Confirm that unsupported questions are refused rather than answered from unrelated chunks.

## 7. Measure deployed latency

```powershell
python scripts/benchmark_deployed_api.py `
  --base-url https://YOUR-PROJECT.vercel.app `
  --repetitions 3
```

This creates:

```text
outputs/deployed_latency_results.json
```

Report local GPU evaluation latency and Vercel production latency separately.

## 8. Final GitHub updates

Replace placeholders in the project and root READMEs with:

- production Vercel URL
- final corpus document/chunk counts
- active retriever and generator models
- measured retrieval and answer metrics
- production latency
- screenshots from the deployed interface

## Troubleshooting

### `transformerReady` is false

The committed metadata still describes hash vectors. Run the MiniLM/E5 embedding step and export again.

### Query retrieval falls back to lexical mode

`HF_API_TOKEN` is missing, invalid, or the configured feature-extraction model is unavailable. Confirm the Vercel environment variable and redeploy.

### Embedding dimension mismatch

Regenerate all document embeddings with one model. Do not edit vectors or dimensions manually.

### Hosted generator fails

The app returns the grounded extractive fallback. Check model/provider availability, token permissions, and function logs. Keep fallback behavior enabled.

### API timeout

Review Vercel function logs, provider latency, and `vercel.json`. Prefer a smaller hosted model or extractive fallback when the provider is slow.

### Wrong directory builds

Set Vercel Project Settings → Build and Deployment → Root Directory to `10-ai-portfolio-rag-assistant`.

### Evaluation panel remains pending

Run `python scripts/build_evaluation_summary.py`, verify the source output files exist, and commit `public/data/evaluation_summary.json`.
