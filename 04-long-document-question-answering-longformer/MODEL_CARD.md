# Model Card — Long-Document Question Answering with Longformer

## Model name

**Portfolio Long-Document QA Pipeline**

## Base checkpoint

`valhalla/longformer-base-4096-finetuned-squadv1`

## Architecture

`LongformerForQuestionAnswering`

## Task

Extractive question answering over long user-provided documents with
overlapping token-window processing, supporting-paragraph selection, and answer
evidence highlighting.

## Training and fine-tuning status

This portfolio project **does not claim to have fine-tuned the checkpoint**.
It loads an existing Longformer checkpoint whose publisher states that it was
fine-tuned on SQuAD v1. The project adds document loading, long-context
windowing, span aggregation, evidence mapping, evaluation, and deployment.

## Intended uses

- Educational demonstrations of Longformer and long-context inference
- Extracting explicit answer spans from public or synthetic reports
- Demonstrating Document AI, evidence localization, Gradio, and Hugging Face
  Spaces deployment
- Prototyping question answering over SOP-like, CAPA-like, supplier-quality, or
  technical documents using safe non-confidential content

## Not intended uses

- Legal, medical, financial, regulatory, safety-critical, or official decisions
- Autonomous quality disposition or CAPA closure
- Processing confidential, proprietary, copyrighted, or personally
  identifiable documents in a public Space
- Questions requiring generation, synthesis across many documents, external
  knowledge, calculations, or answers absent from the document
- Treating the confidence proxy as a calibrated probability of correctness

## Input

- A question
- A document in TXT, Markdown, CSV, PDF with selectable text, or pasted text
- Runtime window length from 512 to 4,096 tokens
- Overlap between consecutive windows

## Output

- Predicted extractive answer span
- Uncalibrated model confidence proxy
- Supporting paragraph
- Highlighted answer evidence
- Paragraph index and document character offsets
- Document length, token-window count, latency, and warnings

## Long-context method

The tokenizer encodes the question and context as paired inputs with
`truncation="only_second"`, overlapping windows, and character offset mappings.
Question tokens receive global attention. The QA head produces start and end
logits for each window. The application selects the highest-scoring valid span
across windows and maps it to the original normalized document.

The checkpoint supports approximately 4,096 tokens in one window. The public
CPU demo defaults to 2,048 tokens per window for better responsiveness.

## Confidence proxy

The displayed value is the geometric mean of the selected start-token and
end-token softmax probabilities within the valid context tokens of that window.
It is not calibrated against real-world answer correctness and can be
overconfident or underconfident.

## Dataset

The repository contains a small synthetic sample dataset with:

- quality CAPA-style report
- supplier quality review
- technical Longformer note

For a larger study, use a documented public long-document QA dataset such as a
QASPER subset, subject to its license and redistribution requirements.

## Evaluation

The project implements:

- Exact Match
- token-level F1
- evidence recall
- context-length analysis
- answer latency
- manual error analysis

Committed results are intentionally marked `not_run`. Run
`python scripts/evaluate_model.py` and
`python scripts/run_context_analysis.py` before publishing any metric values.

## Known limitations

1. The checkpoint was trained on SQuAD-style extractive QA and may not transfer
   reliably to scientific, legal, quality, or technical writing.
2. It does not have a robust learned “no answer” capability.
3. Answers near token-window boundaries can be missed or receive weaker scores.
4. Multiple similar answer spans may produce incorrect evidence selection.
5. PDF extraction does not include OCR.
6. Normalization can shift offsets relative to the original binary document,
   although offsets remain consistent with the text shown in the application.
7. CPU inference over 4,096-token windows can be slow.
8. English is the primary supported language of the checkpoint.

## Bias and risk

Performance can vary by writing style, domain, language, formatting, and entity
type. Users may incorrectly trust highlighted text or confidence values. Every
answer and its evidence must be reviewed by a human.

## Responsible use

This project is for educational and portfolio demonstration only. The model may
produce incomplete, incorrect, unsupported, or misleading answers. Do not
upload private or sensitive documents to the public application, and do not use
outputs as the sole basis for high-impact decisions.

## Deployment

- Gradio entry point: `app.py`
- Hugging Face model source: Hub checkpoint listed above
- Intended Space hardware: CPU for small demonstrations or eligible ZeroGPU
- No training occurs at application startup
