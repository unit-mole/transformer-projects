# Dataset Card — ML Portfolio Documentation Corpus

## Dataset summary

A small public/synthetic knowledge base representing Machine Learning portfolio documentation. It is designed to demonstrate document loading, Markdown-aware preprocessing, section-aware chunking, metadata enrichment, semantic retrieval, and GitHub Pages deployment.

| Field | Description |
|---|---|
| Name | ML Portfolio Documentation Corpus |
| Purpose | Educational semantic-search evaluation and browser demo |
| Format | Markdown source documents and JSON processed artifacts |
| Source directory | `data/raw_documents/` |
| Processed documents | `data/processed/corpus.json` |
| Search chunks | `data/processed/document_chunks.json` |
| Default chunking | Section-aware, approximately 180 words with 40-word overlap |
| License note | Included synthetic text is released with this repository; confirm rights before adding external text |

## Metadata fields

- `chunk_id`
- `document_id`
- `project_name`
- `project_category`
- `source_file`
- `section_title`
- `text`
- `url_or_local_path`
- `tags`
- `document_type`
- `created_from`

## Cleaning and preparation

The pipeline removes YAML frontmatter delimiters, preserves useful frontmatter values as metadata, removes Markdown fence markers while retaining code text, normalizes table separators and whitespace, preserves headings, detects duplicate text, and skips empty documents. It avoids aggressive stemming or stop-word deletion so model names, metrics, deployment platforms, and technical terms remain searchable.

## Safe data policy

The public corpus must not include:

- internal Hach, GCS, quality, complaint, investigation, CAPA, or customer records;
- private emails or attachments;
- personally identifiable information;
- credentials, tokens, or private URLs;
- proprietary source code or documentation;
- copyrighted long-form content without redistribution rights.

Use public repository documentation, self-authored notes, or synthetic examples. Review every file before committing it.

## Known limitations

The included sample is small and intentionally broad. It does not represent production-scale retrieval, all portfolio projects, or real user search behavior. Replace it with the final public portfolio corpus and rerun evaluation before reporting metrics.

## Example record

```json
{
  "chunk_id": "03-cross-encoder-bi-encoder-ranking-system--architecture--000",
  "document_id": "03-cross-encoder-bi-encoder-ranking-system",
  "project_name": "Cross-Encoder and Bi-Encoder Ranking System",
  "project_category": "Transformer / Information Retrieval",
  "source_file": "03-cross-encoder-bi-encoder-ranking-system.md",
  "section_title": "Architecture",
  "text": "A bi-encoder retrieves candidates efficiently and a cross-encoder reranks the shortlist...",
  "url_or_local_path": "https://github.com/unit-mole/transformer-projects/...",
  "tags": ["sentence-transformers", "reranking", "information-retrieval"],
  "document_type": "project_readme",
  "created_from": "public synthetic portfolio documentation"
}
```
