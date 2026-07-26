# Audit of the Provided `Code(8).ipynb`

## What the Original Notebook Did

- 263 total cells: 131 code and 132 markdown.
- Generated 80 synthetic article-summary pairs.
- Loaded 80 real XSum examples when network access was available.
- Implemented Lead and centroid-style extractive baselines.
- Included a DistilBART pipeline function, but set `use_transformer_default` to `False`.
- Calculated token-set F1, compression ratio, a lexical hallucination proxy, latency, and ROUGE for the centroid baseline.
- Exported CSV, JSON, Excel, ZIP, and a generated Streamlit script.

## Key Problems Found

1. **The Transformer was not actually evaluated.** The executed output showed `Transformer sample rows: 0`.
2. **The reported ROUGE values belonged to an extractive centroid baseline, not DistilBART.**
3. **BERTScore was not implemented.**
4. **The requested beam-search and summary-length controls were absent from the production app.**
5. **The app was Streamlit, while the project requirement is Gradio/Hugging Face Spaces.**
6. **Unicode was removed with an ASCII-only regular expression, risking damage to names and multilingual text.**
7. **The method label `transformer_or_fallback` could hide an extractive fallback as a Transformer result.**
8. **There were 75 repeated analysis/checkpoint sections that added volume without new analysis.**
9. **No actual LSTM Seq2Seq comparison was implemented.**
10. **The notebook was a monolith rather than a reusable repository architecture.**

## Improvements in This Rebuild

- Direct, lazy DistilBART inference through `AutoTokenizer` and `AutoModelForSeq2SeqLM`.
- Explicit errors instead of silently presenting a fallback as Transformer output.
- Unicode-safe preprocessing.
- Token-aware long-text map-reduce summarization.
- Full generation controls.
- Gradio app and Hugging Face Space metadata.
- ROUGE, BERTScore, compression, and latency evaluation.
- Baselines and a no-fabrication LSTM comparison framework.
- Modular source files, scripts, tests, CI, Docker, model card, and deployment docs.
- Two concise, professional notebooks instead of repeated filler cells.
