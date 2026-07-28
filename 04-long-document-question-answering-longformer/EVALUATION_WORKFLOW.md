# Project 04 — RTX Training and Evaluation Workflow

This workflow converts Project 04 from a functional demo into a portfolio-grade
Longformer experiment with real, reproducible metrics.

## What will be measured

- Exact Match against all valid contiguous extractive references
- Token-level F1 against all valid references
- Binary evidence recovery at a documented 50% reference-token threshold
- Continuous evidence token recall
- Context-length analysis
- Controlled context-length analysis at approximately 384, 768, 1,536, 3,072,
  and 4,608 tokens
- Answer-position analysis, including answers beyond token 512
- Average, median, and 95th-percentile inference latency
- Throughput
- Peak allocated GPU memory
- Window count
- Confidence-proxy association with Token F1
- Error categories and example-level predictions

## Models compared

1. `deepset/bert-base-cased-squad2` using first-window truncation at 512 tokens
2. `valhalla/longformer-base-4096-finetuned-squadv1` using overlapping windows
3. `models/qasper-longformer/`, produced by continuing Longformer fine-tuning on
   the contiguous extractive QASPER subset

## Step 1 — Open the correct project folder

From Windows Command Prompt:

```bat
cd C:\path\to\transformer-projects\04-long-document-question-answering-longformer
```

## Step 2 — Create and activate a virtual environment

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

## Step 3 — Install CUDA-enabled PyTorch

Use the official PyTorch installation selector for the CUDA version supported by
your NVIDIA driver. Verify that the selected installation is the CUDA build,
not a CPU-only build.

Then install the remaining project dependencies:

```bat
pip install -r requirements.txt -r requirements-evaluation.txt
```

## Step 4 — Verify the RTX GPU

```bat
python scripts\check_gpu.py
```

The output must show:

```text
"cuda_available": true
```

It should also display the GPU name, available VRAM, PyTorch version, CUDA
runtime, and the recommended training profile.

## Step 5 — Start the complete notebook

```bat
jupyter notebook notebooks\complete_longformer_training_evaluation_pipeline.ipynb
```

Use **Kernel → Restart Kernel and Run All Cells** after reviewing the experiment
configuration.

Recommended initial values:

```python
PROFILE = "portfolio"
RUN_FINE_TUNING = True
EVAL_EXAMPLES = 120
LONGFORMER_MAX_LENGTH = 2048
LONGFORMER_STRIDE = 256
```

The notebook uses mixed precision, gradient checkpointing, gradient
accumulation, and automatic batch-size recovery. Do not increase the context
length until the portfolio profile finishes successfully.

## Step 6 — Review generated artifacts

The notebook writes:

```text
outputs/
├── qasper_dataset_summary.json
├── training_summary.json
├── training/training_history.json
├── baseline_comparison.csv
├── baseline_comparison.json
├── controlled_context_length_comparison.csv
├── controlled_context_length_comparison.json
├── evaluation_manifest.json
├── EVALUATION_REPORT.md
├── *_summary.json
├── *_qa_examples.csv
├── *_qa_examples.jsonl
├── *_context_length.csv
├── *_context_length.json
├── *_answer_position.csv
├── *_answer_position.json
├── *_confidence_analysis.json
├── *_error_categories.csv
└── *.png
```

The notebook also replaces marked result sections in:

```text
README.md
MODEL_CARD.md
```

## Step 7 — Perform manual error analysis

Before publishing:

1. Open every `*_qa_examples.csv` file.
2. Inspect at least 20 weak or incorrect examples per model.
3. Confirm that the supporting paragraph actually contains the predicted answer.
4. Review cases where the answer occurs beyond token 512.
5. Compare BERT failures against Longformer recovery.
6. Record any boundary, evidence, ambiguity, and unsupported-span failures.
7. Do not describe the confidence proxy as a calibrated probability.

## Step 8 — Upload the project-fine-tuned checkpoint

Only after verifying the generated results:

```bat
python scripts\push_finetuned_model_to_hub.py --repo-id unit-mole/longformer-qasper-document-qa
```

This uploads the model that was genuinely fine-tuned by the project together
with its evaluation artifacts. Change the repository ID when needed.

## Step 9 — Configure the Gradio Space

To use the fine-tuned model in the Gradio application, set this Space variable:

```text
LONGDOCQA_MODEL_ID=unit-mole/longformer-qasper-document-qa
```

Keep the Static Space clearly labelled as a DistilBERT browser baseline. It does
not run Longformer.

## Step 10 — Push Project 04 to GitHub

Run these commands from the root `transformer-projects` repository, not from the
Project 04 subfolder:

```bat
git add "04-long-document-question-answering-longformer" ".github/workflows/04-long-document-question-answering-longformer.yml"
git commit -m "Add QASPER fine-tuning and benchmark evaluation for Project 04"
git push origin main
```

The `.gitignore` excludes raw QASPER data, processed full datasets, local
checkpoints, model weights, and caches. The generated JSON, CSV, PNG, Markdown,
notebook, source code, tests, and workflow remain eligible for Git.

## Alternative command-line execution

After installing the dependencies, the full workflow can also be run without
opening Jupyter:

```bat
python scripts\run_complete_evaluation.py --profile portfolio --examples 120
```

The notebook remains the recommended portfolio artifact because it presents the
methodology, GPU diagnostics, training, model comparison, generated metrics,
plots, and error-analysis workflow in one reproducible document.
