# Conversion from the provided notebook

## What the provided notebook did

The supplied notebook was titled **“Neural Machine Translation using Transformer Architecture — Unified Synthetic + Real Pipeline,”** but its implemented pipeline was:

- English→French rather than English↔Hindi;
- synthetic phrase generation plus an OPUS Books English–French subset;
- a phrase/word lookup `DictionaryTranslator`, not a trained Transformer inference engine;
- unigram F1 and exact match rather than SacreBLEU and chrF;
- one-way translation;
- repeated analysis cells;
- Streamlit export rather than Gradio;
- no automatic Devanagari/Latin language detection;
- no model-backed confidence proxy;
- no bidirectional MarianMT models;
- no Hugging Face Spaces project structure or lightweight CI.

## What was retained

Useful ideas were retained and generalized:

- Unicode/control-character cleanup;
- data validation;
- deterministic dataset splitting;
- output manifests;
- latency and error-analysis orientation;
- safe Excel/CSV thinking;
- a clear distinction between sample and public data.

## What was rebuilt

The production path was rebuilt around:

- actual English→Hindi and Hindi→English MarianMT models;
- direct `AutoTokenizer` and `AutoModelForSeq2SeqLM` loading;
- lazy model caching;
- shared sentence and batch inference;
- script-based automatic direction detection;
- confidence-proxy scoring from generation outputs;
- SacreBLEU, chrF, and latency evaluation;
- Gradio and Hugging Face Spaces;
- unit tests, CI, model metadata, deployment docs, and responsible-use language.

The original notebook is preserved under `notebooks/archive/` for traceability and should not be presented as the final model implementation.
