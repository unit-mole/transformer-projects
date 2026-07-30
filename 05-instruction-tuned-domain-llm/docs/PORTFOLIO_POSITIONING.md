# Portfolio Positioning

## One-line project description

Fine-tuned FLAN-T5-base with LoRA/PEFT on a reviewed custom ML/Data Science instruction dataset, measured it against the base model on an independent benchmark, and deployed the adapter through a responsible Gradio Hugging Face Space.

## GitHub pinned-repository description

End-to-end instruction-tuned domain LLM project featuring RTX-optimized FLAN-T5-base LoRA training, a reviewed custom ML/DS dataset, an 80-prompt held-out benchmark, multi-metric base-vs-LoRA evaluation, human review, and Hugging Face deployment.

## Resume bullets — use only after the experiment is completed

- Built an instruction-tuned ML/Data Science learning assistant using the FLAN-T5-base encoder-decoder Transformer and LoRA/PEFT, including hardware-aware mixed-precision training, early stopping, adapter export, and deployment-safe inference.
- Created a custom ML/DS instruction-data workflow targeting approximately 600 reviewed examples, with schema validation, PII/confidential-data checks, near-duplicate removal, benchmark-leakage screening, and stratified splits.
- Evaluated base and LoRA models on the same 80-example held-out benchmark using BERTScore, ROUGE-L, sentence-embedding similarity, instruction adherence, response-quality rubrics, latency, hallucination-risk review, category slices, and bootstrap confidence intervals.
- Deployed the reviewed PEFT adapter through a Gradio Hugging Face Space and added reproducibility metadata, model/dataset cards, automated tests, GitHub Actions, and evidence-based portfolio-readiness checks.

Do not use these bullets before the corresponding adapter, metrics, review, and deployment artifacts exist.

## Skills demonstrated

Transformer models, encoder-decoder architecture, instruction tuning, FLAN-T5, LoRA, PEFT, local synthetic-data generation, dataset governance, leakage prevention, mixed-precision GPU training, early stopping, seq2seq evaluation, BERTScore, ROUGE-L, embedding similarity, bootstrap confidence intervals, hallucination analysis, human evaluation, responsible AI, Gradio, Hugging Face Hub/Spaces, GitHub Actions, testing, and reproducibility.

## Screenshots to include

1. Notebook GPU/hardware report.
2. Dataset distribution and validation report.
3. Real training and validation loss curve.
4. Base-versus-LoRA metric comparison chart.
5. Category-level comparison table.
6. A strong before/after response example.
7. A regression or failure-analysis example showing honest evaluation.
8. Gradio response and inference metadata showing `lora_adapter`.
9. Hugging Face adapter model card.
10. Live Hugging Face Space page.

## Connection to Quality Data Science

The assistant is relevant to quality analytics because it demonstrates how an AI system can explain model choices, compare classifiers and anomaly-detection methods, generate non-confidential training examples, support technical onboarding, and prototype future internal learning tools. The public project uses generic examples and keeps human experts responsible for engineering and quality conclusions.
