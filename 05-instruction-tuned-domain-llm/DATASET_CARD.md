# Dataset Card — ML/Data Science Instruction Dataset V2

## Dataset Name

`ml_ds_instruction_dataset_v2`

## Purpose

Create a reviewed, redistributable, domain-focused instruction corpus for adapting FLAN-T5-base into an ML and Data Science Learning Assistant.

## Data Assets

| File | Purpose | Size before notebook execution |
|---|---|---:|
| `ml_ds_instruction_dataset.jsonl` | Original self-authored seed data | 93 |
| `dataset_generation_plan.json` | 64-topic synthetic-generation plan | 64 topics |
| `ml_ds_instruction_dataset_v2.jsonl` | Expanded and reviewed training corpus | Generated locally, target about 600 |
| `benchmark_prompts_v2.jsonl` | Independent reference benchmark | 80 |

The benchmark is not used for training or candidate generation prompts.

## Format

Each JSONL training record contains:

- `id`
- `instruction`
- `input`
- `output`
- `category`
- `difficulty`
- `topic`
- `source`
- `split`

The benchmark additionally includes `reference_answer` and uses `split: benchmark`.

## Instruction Categories

Concept explanation, algorithm comparison, metric explanation, example generation, beginner-friendly explanation, interview-style answer, small code example, Data Science workflow, ML project guidance, and quality analytics.

## Topic Coverage

The generation plan covers 64 topics across fundamentals, preprocessing, supervised and unsupervised algorithms, validation, classification and regression metrics, deep learning, Transformers, instruction tuning, LoRA/PEFT, RAG, hallucination analysis, and quality analytics.

## Creation Process

1. Load the 93 self-authored seed records.
2. Use a compact local instruction model to draft diverse candidate records for the 64-topic plan.
3. Normalize the candidate schema and source labels.
4. Remove empty, too-short, too-long, PII-like, confidential, and invalid records.
5. Remove exact and TF-IDF near-duplicate instructions.
6. Compare candidate instructions with the held-out benchmark and remove likely leakage.
7. Assign category-stratified train, validation, and internal test splits.
8. Save raw generations, errors/retries, duplicate removals, leakage removals, and a quality report.
9. Require human review before model training.

The local teacher model drafts candidate data; it is not the final deployed model.

## Quality Targets

The notebook requires at least:

- 450 validated examples overall;
- 40 validation examples;
- 40 internal test examples;
- a separate 80-example benchmark;
- review samples from every category;
- review of all advanced and code examples;
- no benchmark-like training instructions above the configured similarity threshold.

## Cleaning and Validation

The pipeline checks:

- missing instructions and outputs;
- duplicate and near-duplicate instructions;
- overly short and overly long answers;
- invalid categories, difficulties, or split labels;
- email and phone patterns that may indicate PII;
- confidential or proprietary-data terms;
- similarity to benchmark instructions;
- category, difficulty, topic, prompt-length, and answer-length distributions.

## Human Review

Synthetic candidate data is not considered approved automatically. Before training, review:

- at least two random examples per category;
- every advanced example;
- every code example;
- any record flagged by validation;
- factual definitions, formulas, comparisons, caveats, and code behavior.

Correct or remove weak records and rerun validation.

## Known Limitations

- Candidate records can inherit errors or style bias from the local teacher model.
- Topic coverage is broad but not equivalent to a complete curriculum.
- Automated de-duplication and leakage thresholds are imperfect.
- Self-authored benchmark answers can reflect author preferences.
- Human review improves quality but does not guarantee error-free data.

## Sensitive Data Handling

Do not add names, addresses, emails, phone numbers, customer identifiers, confidential cases, proprietary process details, restricted course material, or copyrighted textbook passages. Use generic or fictional quality analytics examples.

## License and Usage Note

The self-authored seed records, generation plan, and benchmark are provided under the repository license. Review the license and usage conditions of the local teacher model and the FLAN-T5 base model separately.

## Responsible Use

Training on this dataset does not create a verified expert. Generated answers require human review and must remain within the educational ML/Data Science scope.
