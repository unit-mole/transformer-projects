# Model Card — Browser Deployment Baseline

## Model

```text
Xenova/distilbert-base-cased-distilled-squad
```

## Role in this project

This model powers only the free Hugging Face Static Space. It is a
browser-compatible ONNX extractive question-answering model used after
client-side long-document chunking and lexical candidate retrieval.

It is **not** the Longformer model used by the full Python implementation.

## Core Python model

```text
valhalla/longformer-base-4096-finetuned-squadv1
```

## Task

Extract an answer span from a selected context chunk given a user question.

## Long-document adaptation

Because DistilBERT has a shorter context window, the browser application:

1. creates overlapping word chunks;
2. ranks chunks against the question;
3. evaluates the highest-ranked chunks;
4. chooses the strongest valid answer span;
5. maps it to a supporting paragraph;
6. highlights the evidence.

## Intended use

- educational demonstrations;
- portfolio review;
- non-sensitive report or policy exploration;
- comparison with the Longformer Python pipeline.

## Not intended use

- high-stakes decision-making;
- confidential or proprietary documents;
- medical, legal, financial, regulated, or safety-critical advice;
- claims that the browser result came from Longformer;
- claims that this checkpoint was trained by the portfolio author.

## Metrics

No browser-baseline performance metrics are claimed until the project evaluation
is actually run. The interface displays per-request latency and model confidence
proxy only.

## Limitations

- retrieval may omit the correct chunk;
- answer quality depends on explicit evidence;
- the score is not calibrated;
- short-context chunking can lose cross-paragraph relationships;
- PDF extraction supports selectable text, not OCR;
- first-load latency depends on model download and browser hardware.
