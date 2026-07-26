# Dataset Card — ML/DS Instruction Curriculum

## Dataset summary

A compact custom dataset of **82** self-authored instruction-response examples for adapting a small instruction model into an ML and Data Science Learning Assistant.

## Purpose

The dataset supports concept explanations, algorithm comparisons, metric explanations, code examples, interview-style answers, workflow guidance, quality analytics scenarios, and responsible-AI guidance.

## Format

JSON Lines with `instruction`, optional `input`, `output`, category, difficulty, topic, source, reference answer, identifier, and split.

## Data sources

All examples are newly authored synthetic curriculum content. No confidential quality reports, personal data, proprietary company text, or copyrighted textbook excerpts are included.

## Cleaning and validation

The preparation script checks empty required fields, short outputs, duplicate prompt/input combinations, JSON validity, and simple patterns that may indicate sensitive data. Technical correctness still requires human review.

## Splits

A deterministic pattern assigns approximately 80% training, 10% validation, and 10% test examples.

## Known limitations

- Small size and limited linguistic diversity.
- Primarily English-language content.
- Reference answers are concise and do not represent every valid response.
- Code examples require execution and version-specific review.
- Synthetic examples may not match real learner phrasing.

## License and use

Dataset content is released with this repository under the MIT License for educational and portfolio use. Users remain responsible for reviewing generated outputs and third-party model licenses.

## Responsible use

Do not add private, confidential, proprietary, copyrighted, or personally identifiable material. Do not use this dataset or model as a high-stakes advisor.

## Example

```json
{
  "instruction": "Explain supervised learning in simple terms.",
  "input": "",
  "output": "Supervised learning trains a model from labeled examples, where each input is paired with a known target. The model learns a mapping that can predict targets for new data. Common tasks are classification and regression.",
  "response": "Supervised learning trains a model from labeled examples, where each input is paired with a known target. The model learns a mapping that can predict targets for new data. Common tasks are classification and regression.",
  "category": "concept_explanation",
  "difficulty": "beginner",
  "topic": "supervised learning",
  "source": "self-authored synthetic curriculum",
  "reference_answer": "Supervised learning trains a model from labeled examples, where each input is paired with a known target. The model learns a mapping that can predict targets for new data. Common tasks are classification and regression.",
  "id": "mlds-0001",
  "split": "test"
}
```
