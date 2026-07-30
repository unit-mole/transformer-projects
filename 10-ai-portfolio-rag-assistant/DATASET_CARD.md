# Dataset Card — Public AI Portfolio Corpus

## Dataset name

**Anmol Tripathi Public AI Portfolio Corpus**

## Purpose

Provide a traceable knowledge base for semantic retrieval and source-cited RAG answers about public ANN, RNN, LSTM, BiLSTM, CNN, and Transformer projects.

## Source repositories

The collection configuration is stored in `config/portfolio_repositories.json`. Repository names must be verified before collection. Only public GitHub Markdown is eligible.

## Document types

- `README.md`
- `MODEL_CARD.md`
- `DATASET_CARD.md`
- `README_HUGGINGFACE.md`
- `README_GITHUB_PAGES.md`
- `README_VERCEL.md`
- other explicitly approved public deployment/project READMEs

## Format

| Stage | Location |
|---|---|
| Raw public Markdown | `data/raw_portfolio_docs/` |
| Processed documents | `data/processed/portfolio_corpus.json` |
| Section-aware chunks | `data/processed/document_chunks.json` |
| Normalized embeddings | `data/processed/embeddings.json` |
| Evaluation questions | `data/processed/evaluation_questions.json` |
| Vercel runtime artifacts | `public/data/*.json` |

## Coverage

The checked-in starter corpus is not the final dataset. The final metadata must report all six portfolio categories:

```text
ANN
Simple RNN
LSTM
BiLSTM
CNN
Transformer
```

Final document and chunk counts are generated automatically and must not be invented.

## Cleaning and preprocessing

- UTF-8 decoding with replacement for malformed characters.
- Frontmatter and HTML-comment cleanup.
- Whitespace normalization.
- Heading and section preservation.
- Technical model/dataset/metric/deployment terms preserved.
- Empty documents skipped.
- SHA-256 duplicate detection.
- Safe JSON export.

## Chunking

Default strategy:

```text
section-aware Markdown splitting
chunk size: 220 words
chunk overlap: 50 words
```

Each chunk stores its heading and word boundaries. Settings are configurable and should be evaluated rather than assumed optimal.

## Metadata fields

- `id`
- `documentId`
- `projectId`
- `projectName`
- `category`
- `deployment`
- `repository`
- `repositoryUrl`
- `sourceFile`
- `section`
- `sourcePath`
- `checksumSha256`
- `startWord`
- `endWord`
- `keywords`
- `text`

## Evaluation set

The included evaluation set contains at least 40 curated questions covering direct facts, paraphrases, multi-project retrieval, deployment, skills, ambiguous queries, and unsupported/private-information requests. Answerable questions include expected project IDs; unsupported questions are labeled `answerable: false`.

## Sensitive-data policy

Do not include:

- Hach or other employer-internal files
- GCS cases or quality-case histories
- corrective-action records
- customer or supplier data
- private email
- API keys, credentials, or tokens
- unpublished proprietary documentation
- PII beyond public portfolio identity
- copyrighted material without redistribution rights

The collector is designed for public Markdown, but the human maintainer remains responsible for reviewing every source before commit.

## Known limitations

- Public README content may be incomplete or stale.
- Repository naming and document quality vary.
- Deployment descriptions may not reflect current availability.
- Expected evaluation labels require manual verification.
- Similar projects can share terminology and produce retrieval ambiguity.
- The corpus is a portfolio dataset, not a general-purpose ML knowledge base.

## License and usage

Project code is MIT-licensed. Each source repository, model, dataset, and document remains subject to its own license and terms. Public availability does not automatically grant unrestricted redistribution rights.
