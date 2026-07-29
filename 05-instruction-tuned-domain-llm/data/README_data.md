# ML/Data Science Instruction Dataset

This project includes **82 self-authored, synthetic curriculum examples** created specifically for a public portfolio demonstration. It contains no private company cases, personally identifiable information, proprietary course material, or copied textbook passages.

## Schema

| Field | Meaning |
|---|---|
| `id` | Stable example identifier |
| `instruction` | Natural-language learning task |
| `input` | Optional context |
| `output` / `response` | Target answer |
| `reference_answer` | Reference used for evaluation |
| `category` | Capability group |
| `difficulty` | Beginner or intermediate |
| `topic` | ML/DS topic |
| `source` | Provenance note |
| `split` | Train, validation, or test |

## Current statistics

- Examples: **82**
- Topic count: **77**
- Average prompt length: **6.6 words**
- Average response length: **35.9 words**
- Split strategy: deterministic 80/10/10 pattern

## Limitations

The dataset is intentionally compact. It is suitable for validating the training and deployment workflow, but a stronger production model would require a larger expert-reviewed corpus, broader phrasing, adversarial prompts, code execution checks, and more rigorous factual validation.


## Expanded portfolio dataset

Run `python scripts/build_extended_dataset.py` to rebuild `ml_ds_instruction_dataset_extended.jsonl` and `evaluation_prompts_extended.jsonl`. The generated dataset contains 401 public-safe examples with topic-grouped 323/42/36 train, validation, and test splits. `topic_group` prevents paraphrases of the same concept from crossing split boundaries. Generated statistics and validation results are stored in `outputs/extended_dataset_statistics.json` and `outputs/extended_dataset_validation_report.json`.
