# Local RTX Full Experiment Guide

This guide takes Project 05 from the current untrained scaffold to a reviewed FLAN-T5-base LoRA release candidate with real metrics.

## 1. Use an isolated Python environment

On Windows PowerShell from the Project 05 directory:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip wheel setuptools
```

Python 3.12 is recommended for broad compatibility across PyTorch, Transformers, PEFT, sentence-transformers, Jupyter, and evaluation libraries. Do not reuse the environment of another portfolio project.

## 2. Verify NVIDIA driver and GPU

```powershell
nvidia-smi
```

Record the GPU name, driver version, and reported VRAM. The notebook also saves these details automatically.

## 3. Install a CUDA-enabled PyTorch build

Use the official PyTorch installation selector for:

- OS: Windows
- Package: Pip
- Language: Python
- Compute platform: the CUDA option supported by the current driver

Run the command produced by the selector inside `.venv`. Then verify:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

Continue only when `torch.cuda.is_available()` prints `True`.

## 4. Install the project environment

```powershell
python -m pip install -r requirements-training.txt
python -m ipykernel install --user --name project05-flan-t5 --display-name "Project 05 FLAN-T5 LoRA"
pytest -q
```

All tests should pass before model execution.

## 5. Start Jupyter

```powershell
jupyter lab
```

Open:

```text
notebooks/05_full_training_evaluation_pipeline.ipynb
```

Select the `Project 05 FLAN-T5 LoRA` kernel.

## 6. Run the environment and benchmark cells

Run Sections 0–4. Confirm:

- CUDA is available;
- the correct RTX GPU is displayed;
- BF16 or FP16 is selected;
- `google/flan-t5-base` is the recommended model;
- the benchmark contains 80 records.

If the notebook recommends `flan-t5-small` because VRAM is below the quality threshold, you can still force the base model, but reduce the micro-batch to 1 and keep gradient accumulation/checkpointing enabled.

## 7. Generate the enhanced dataset

Run the dataset-generation cell with:

```python
TARGET_DATASET_EXAMPLES = 600
TEACHER_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
```

The stage creates:

```text
data/ml_ds_instruction_dataset_v2.jsonl
outputs/experiments/<run>/dataset_generation/raw_teacher_generations.jsonl
outputs/experiments/<run>/dataset_generation/teacher_generation_log.json
outputs/experiments/<run>/dataset_generation/enhanced_dataset_quality_report.json
outputs/experiments/<run>/dataset_generation/duplicate_removals.json
outputs/experiments/<run>/dataset_generation/benchmark_leakage_removals.json
```

The final validated corpus must contain at least 450 records, at least 40 validation records, and at least 40 internal test records.

## 8. Perform the dataset review

Review:

- at least two examples from every category;
- every advanced record;
- every small-code record;
- records with formulas, numeric examples, or strong claims;
- records where the answer does not match the requested format.

Correct or delete weak records directly in `ml_ds_instruction_dataset_v2.jsonl`, rerun validation, and only then set:

```python
DATASET_HUMAN_REVIEW_APPROVED = True
```

This approval is a portfolio-quality control, not an inconvenience.

## 9. Train the LoRA adapter

Use the quality configuration:

```text
Base model: google/flan-t5-base
LoRA rank: 16
LoRA alpha: 32
LoRA dropout: 0.05
Learning rate: 1e-4
Maximum epochs: 6
Early stopping patience: 2
Scheduler: cosine
```

The notebook automatically selects BF16/FP16, micro-batch size, gradient accumulation, and checkpointing from GPU capability.

Successful training creates:

```text
outputs/experiments/<run>/training/lora_adapter/adapter_config.json
outputs/experiments/<run>/training/lora_adapter/adapter_model.safetensors
outputs/experiments/<run>/training/tokenizer/
outputs/experiments/<run>/training/model_metadata.json
outputs/experiments/<run>/training/hardware_report.json
outputs/experiments/<run>/training/training_log_history.json
outputs/experiments/<run>/training/training_log_history.csv
outputs/experiments/<run>/training/training_curve.png
```

Inspect the training and validation curves. A falling training loss with worsening validation loss indicates overfitting; early stopping should limit it.

## 10. Run base-versus-LoRA evaluation

The notebook loads and evaluates:

1. base `google/flan-t5-base`;
2. the trained LoRA adapter;
3. the same 80 held-out prompts and deterministic generation settings.

It saves:

```text
outputs/experiments/<run>/evaluation/base_model/metrics.json
outputs/experiments/<run>/evaluation/lora_model/metrics.json
outputs/experiments/<run>/evaluation/comparison/base_vs_lora_comparison.json
outputs/experiments/<run>/evaluation/comparison/per_example_base_vs_lora.csv
outputs/experiments/<run>/evaluation/comparison/base_vs_lora_metric_comparison.png
outputs/experiments/<run>/evaluation/comparison/before_after_finetuning_examples.md
```

The comparison includes BERTScore, ROUGE-L, semantic similarity, adherence, response-quality rubric, latency, hallucination-risk flags, win rates, category slices, and bootstrap confidence intervals.

## 11. Complete response review

Review all:

- heuristic hallucination flags;
- code answers;
- advanced prompts;
- negative LoRA deltas;
- random examples from each category.

Fill the manual-rating fields in the CSVs. Do not present automated metrics as proof of factual correctness.

## 12. Promote reviewed artifacts

After review, set:

```python
EVALUATION_HUMAN_REVIEW_COMPLETED = True
PROMOTE_REVIEWED_ARTIFACTS = True
```

Promotion copies the final adapter and selected evidence files into `models/` and `outputs/`, creates SHA-256 checksums, and writes `outputs/release_manifest.json`.

## 13. Check the 9/10 evidence target

```powershell
python scripts/check_portfolio_readiness.py
```

The report is saved as:

```text
outputs/portfolio_readiness_report.json
```

This score measures whether the project contains the expected evidence. It is not a universal measure of the model's intelligence.

## 14. Test the Gradio app

```powershell
python app.py
```

Ask several benchmark and new questions. Confirm the metadata reports:

```text
model_mode: lora_adapter
```

Do not deploy while it reports `base_model_fallback`.

## 15. Publish the adapter and Space

Push the compact adapter to a Hugging Face model repository. In the Space settings, set:

```text
BASE_MODEL_ID=google/flan-t5-base
ADAPTER_ID=<username>/flan-t5-base-ml-ds-lora
```

Add the final model and Space links to `README.md` and `MODEL_CARD.md`, then commit only reviewed outputs.
