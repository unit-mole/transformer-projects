# Data Documentation

## Included sample

The repository contains a public-safe synthetic ranking dataset created only for
portfolio demonstration.

| File | Key columns |
|---|---|
| `sample_documents.csv` | `document_id`, `title`, `document`, `category`, `source` |
| `sample_queries.csv` | `query_id`, `query`, `split`, `domain` |
| `sample_qrels.csv` | `query_id`, `document_id`, `relevance` |
| `sample_job_search_data.csv` | `document_id`, `job_title`, `job_description`, `source` |

Relevance is graded from 1 to 3:

- 3: highly relevant;
- 2: relevant;
- 1: partially relevant;
- 0 or missing: not labeled relevant.

## Safety

Do not commit:

- private GCS or complaint narratives;
- confidential company documents;
- proprietary job descriptions;
- resumes containing personal information;
- restricted or copyrighted datasets without permission;
- large raw datasets or generated caches.

Use `data/raw/` or `data/private/` locally; both are ignored by Git.

## Adapting to MS MARCO

For a public benchmark version:

1. download an appropriately licensed MS MARCO subset outside Git;
2. map passages to `document_id`, `title`, and `document`;
3. map queries to `query_id` and `query`;
4. map relevance judgments to `query_id`, `document_id`, and `relevance`;
5. document the source, license, split, and row counts;
6. run evaluation and replace only the generated output files.

The repository does not bundle MS MARCO data.
