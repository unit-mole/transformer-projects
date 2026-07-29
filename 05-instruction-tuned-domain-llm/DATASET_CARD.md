# Dataset Card — ML/DS Instruction Curriculum

## Dataset summary

A public-safe, self-authored and curated curriculum containing **401 instruction-response examples** for adapting a small instruction model into an ML and Data Science Learning Assistant. The original 82-example seed dataset is preserved, while `ml_ds_instruction_dataset_extended.jsonl` is the recommended training dataset.

## Purpose

The dataset supports concept explanations, algorithm comparisons, metric explanations, code examples, interview-style answers, workflow guidance, quality analytics scenarios, and responsible-AI guidance.

## Format

JSON Lines with `instruction`, optional `input`, `output`, `response`, category, difficulty, topic, `topic_group`, source, reference answer, identifier, and split.

## Statistics

| Statistic | Value |
|---|---:|
| Examples | 401 |
| Train / validation / test | 323 / 42 / 36 |
| Capability categories | 9 |
| Topic groups | 203 |
| Unique topics | 205 |
| Average prompt words | 7.79 |
| Average response words | 43.84 |

The exact generated statistics are saved in `outputs/extended_dataset_statistics.json`.

## Data sources

All examples are newly authored or curated for this project. No confidential quality reports, personal data, proprietary company text, or copied textbook passages are included. Quality-analytics examples are generic and synthetic.

## Creation process

`scripts/build_extended_dataset.py` combines the original seed records with a structured curriculum covering classical ML, deep learning, Transformers, instruction tuning, LoRA, evaluation, deployment, MLOps, quality analytics, and responsible AI. The generator creates multiple task styles only when the responses can be grounded in a curated concept record.

## Cleaning and validation

The project checks:

- valid JSONL records;
- required fields;
- duplicate prompt/input pairs;
- minimum and maximum response length;
- simple sensitive-data patterns;
- valid train, validation, and test labels;
- topic-group isolation across splits.

The generated validation report currently contains zero structural issues. Technical correctness still requires human review.

## Splits

Splits are assigned deterministically at the **topic-group level**, not at the individual-row level. Therefore, paraphrases and prompt variants for one topic cannot appear in both training and test sets. The target allocation is approximately 80% training, 10% validation, and 10% test.

## Known limitations

- The curriculum is synthetic and primarily English.
- Reference answers are concise and do not represent every valid answer.
- Some related concepts remain semantically close even when topic groups differ.
- Code examples require execution and package-version review.
- The dataset is sufficient for a portfolio-scale LoRA experiment, not a production-grade educational model.
- Expert review should be expanded before high-stakes or broad public use.

## License and use

Dataset content is released with this repository under the MIT License for educational and portfolio use. Users remain responsible for reviewing generated outputs and complying with third-party model licenses.

## Responsible use

Do not add private, confidential, proprietary, copyrighted, or personally identifiable material. Do not use this dataset or resulting model as a legal, medical, financial, immigration, safety-critical, or official advisor.

## Example

```json
{
  "instruction": "Explain LoRA in simple terms.",
  "input": "",
  "output": "LoRA adds trainable low-rank matrices to selected layers while leaving base weights frozen...",
  "category": "concept_explanation",
  "difficulty": "beginner",
  "topic": "LoRA",
  "topic_group": "lora",
  "source": "self-authored and curated public-safe ML/DS curriculum",
  "reference_answer": "LoRA adds trainable low-rank matrices to selected layers while leaving base weights frozen...",
  "id": "mlds-ext-...",
  "split": "train"
}
```
