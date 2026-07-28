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

## QASPER evaluation data

The portfolio-grade evaluation workflow downloads QASPER v0.3 from the official
AllenAI-hosted archives at runtime. It retains only answerable questions with a
contiguous extractive span found in the reconstructed paper text. Free-form,
yes/no, unanswerable, unresolved, and non-contiguous multi-span annotations are
not silently converted into span labels.

Generated raw and processed QASPER files are excluded from Git. Commit only the
small dataset summary and model-evaluation artifacts created under `outputs/`.
QASPER is documented as CC BY 4.0 in its dataset card; retain attribution in
portfolio documentation.
