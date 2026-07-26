# Data documentation

The repository includes only **small synthetic sample documents** created for
portfolio demonstration. They contain no employer, customer, patient,
proprietary, or personally identifiable information.

## Included files

- `sample_documents/quality_capa_report.txt`
- `sample_documents/supplier_quality_report.txt`
- `sample_documents/longformer_overview.md`
- `sample_questions.csv`
- `sample_qa_pairs.csv`

## Evaluation schema

| Column | Meaning |
|---|---|
| `example_id` | Stable sample identifier |
| `document_name` | File under `sample_documents/` |
| `question` | Extractive QA question |
| `answer` | Reference answer text |
| `reference_evidence` | Human-written evidence passage |
| `reference_paragraph_index` | Expected paragraph/section index |
| `source_type` | Provenance category |

## Optional public dataset

For a larger experiment, use a documented subset of QASPER or another
redistributable long-document QA dataset through a separate download script.
Do not commit the full dataset or restricted documents to GitHub.

## Data safety

Never upload confidential quality reports, CAPA records, complaint
investigations, SOPs, technical manuals, copyrighted reports, or documents
containing personally identifiable information to a public Hugging Face Space.
