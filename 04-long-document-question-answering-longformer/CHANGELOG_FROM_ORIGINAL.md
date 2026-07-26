# Review of the supplied notebook and implemented changes

## What the supplied notebook actually did

The uploaded notebook, **LongDocQA 360**, created synthetic QA examples, loaded
a small SQuAD validation subset when internet access was available, repeated
SQuAD contexts to imitate longer documents, split text into fixed word chunks,
ranked chunks with TF-IDF, and answered with either:

- a sentence-overlap heuristic, which was the default because
  `USE_TRANSFORMER_QA = False`; or
- `distilbert-base-cased-distilled-squad` when the optional transformer switch
  was enabled.

It generated CSV/Excel/ZIP outputs and wrote a Streamlit application.

## Technical gaps found

1. The notebook title said “Longformer Style,” but it did not load a Longformer
   or BigBird checkpoint.
2. The default backend was a lexical sentence heuristic, not Transformer QA.
3. Repeating SQuAD paragraphs does not create a genuine long-document QA
   benchmark.
4. Chunking used word counts rather than tokenizer-aware overlapping windows.
5. The application was Streamlit, while this portfolio project requires Gradio
   and Hugging Face Spaces.
6. It did not upload or parse TXT, Markdown, CSV, and PDF documents.
7. It did not map answer-token offsets back to the original document.
8. It did not return a supporting paragraph and highlighted evidence.
9. The score was absent for the default backend, and confidence calibration was
   not explained.
10. “Grounding overlap” was not the requested evidence-recall metric.
11. Context-length buckets and per-bucket QA performance were not implemented.
12. Cells 54–115 repeated similar checkpoint analyses and increased notebook
    length without adding distinct modeling capability.
13. No modular package, unit tests, model card, GitHub Actions workflow, or
    Hugging Face deployment structure was present.

## What this version changes

- Uses `valhalla/longformer-base-4096-finetuned-squadv1` through
  `AutoModelForQuestionAnswering`.
- Performs tokenizer-aware sliding-window inference and evaluates answer spans
  across every window.
- Applies global attention to question tokens for Longformer QA.
- Supports TXT, Markdown, CSV, and selectable-text PDF files.
- Maps answer offsets to the normalized source document.
- Returns answer, confidence proxy, supporting paragraph, paragraph index,
  highlighted evidence, latency, token-window count, and warnings.
- Adds Exact Match, token-level F1, evidence recall, context-length analysis,
  latency, and manual error-analysis scripts.
- Replaces Streamlit with a polished Gradio application.
- Adds sample documents that are synthetic and safe to publish.
- Adds tests, CI, Docker, model metadata, a model card, and deployment guides.
- Preserves the original notebook under `notebooks/archive/` for traceability.

## Claims intentionally not made

This project does **not** claim that Anmol fine-tuned the selected checkpoint.
The checkpoint is loaded from Hugging Face Hub and was already fine-tuned on
SQuAD v1 by its publisher. Metric placeholders remain `null` until the supplied
evaluation scripts are run.
