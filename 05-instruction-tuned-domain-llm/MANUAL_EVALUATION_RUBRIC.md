# Manual Evaluation Rubric

Automated metrics cannot establish whether every educational answer is factually correct. Review a stratified sample from `outputs/portfolio_experiment/manual_review_results.csv`.

## Rating scale

| Score | Correctness | Relevance | Clarity |
|---:|---|---|---|
| 5 | Correct and complete for the requested scope | Directly answers every important part | Clear, well structured, and appropriately concise |
| 4 | Correct with a minor omission or caveat missing | Answers the task with small irrelevant detail | Clear with minor wording issues |
| 3 | Partly correct or noticeably incomplete | Addresses the topic but misses a requirement | Understandable but confusing or poorly organized in places |
| 2 | Contains a major technical error | Mostly generic or off-target | Difficult to follow or misleading |
| 1 | Incorrect, fabricated, unsafe, or unusable | Does not answer the instruction | Incoherent or seriously misleading |

## Hallucination label

Mark `yes` when the answer contains an unsupported definition, invented fact, incorrect formula, fabricated attribution, unsupported number, wrong code behavior, or confident claim not justified by the reference and trusted technical knowledge.

Mark `no` only after reviewing the full answer. Mark `uncertain` when external verification is needed.

## Preferred model

Choose one of:

- `base`
- `lora`
- `tie`
- `neither`

Do not select the LoRA response merely because it is longer or uses more technical vocabulary.

## Reviewer notes

Record the precise reason for low scores. Useful notes include:

- missing definition;
- incorrect metric interpretation;
- compares only one algorithm;
- code has an undefined variable;
- omits leakage warning;
- too generic for the requested quality-analytics example;
- valid response but different from the reference wording.

## Recommended review quality

Review at least 20–30 prompts across categories. For an even stronger portfolio claim, use a second reviewer on a subset and report disagreements or adjudication notes.
