---
language:
- en
license: apache-2.0
library_name: peft
pipeline_tag: text2text-generation
base_model: google/flan-t5-base
tags:
- transformers
- flan-t5
- peft
- lora
- instruction-tuning
- machine-learning
- data-science
- educational
---

# Model Card — FLAN-T5-base ML/Data Science LoRA Assistant

## Model Name

`flan-t5-base-ml-ds-lora` — replace this working name with the final Hugging Face adapter repository ID after publication.

## Current Status

The repository contains the complete training, evaluation, review, and deployment pipeline. A trained adapter and numeric results are considered final only after `notebooks/05_full_training_evaluation_pipeline.ipynb` is executed, human review is completed, and the experiment is promoted through `src/release_utils.py`.

Do not publish placeholder or `not_run` values as model results.

## Base Model

- Model: `google/flan-t5-base`
- Architecture: encoder-decoder Transformer
- Task family: text-to-text sequence generation
- Base-model license: Apache-2.0

`google/flan-t5-small` remains an optional lower-memory fallback, but the portfolio-quality experiment is designed around FLAN-T5-base.

## Fine-Tuning Method

Parameter-efficient fine-tuning with Hugging Face PEFT LoRA:

| Parameter | Quality preset |
|---|---:|
| Task type | `SEQ_2_SEQ_LM` |
| Target modules | `q`, `v` |
| Rank | 16 |
| Alpha | 32 |
| Dropout | 0.05 |
| Maximum epochs | 6 |
| Learning rate | `1e-4` |
| Scheduler | cosine |
| Label smoothing | 0.05 |
| Early stopping patience | 2 validation evaluations |
| Precision | BF16 when supported, otherwise FP16 |

Only the adapter parameters are trained. This is not full-model fine-tuning.

## Task

Follow educational instructions about Machine Learning and Data Science, including:

- concept and metric explanation;
- algorithm comparison;
- beginner-friendly and interview-style answers;
- small Python or pseudocode examples;
- Data Science workflows and project guidance;
- non-confidential quality analytics examples.

## Training Data

The workflow starts with 93 self-authored seed records and locally generates candidate records toward a target of approximately 600 examples. Candidate records are validated, near-deduplicated, checked for similarity against the held-out benchmark, split by category, and manually reviewed before training.

The 80-example benchmark is self-authored and excluded from training. See `DATASET_CARD.md`.

## Training and Reproducibility Artifacts

A completed run saves:

- `adapter_model.safetensors` and `adapter_config.json`;
- tokenizer files;
- hardware and CUDA report;
- package versions and random seed;
- exact LoRA/training configuration;
- train, validation, and internal test metrics;
- validation perplexity;
- trainer state and best checkpoint;
- JSON/CSV log history and a real training curve.

## Evaluation Protocol

Base FLAN-T5 and LoRA FLAN-T5 answer the same 80 held-out prompts with deterministic decoding. The evaluation includes:

- instruction-adherence heuristic;
- transparent response-quality rubric;
- BERTScore precision, recall, and F1;
- ROUGE-L F1;
- sentence-embedding semantic similarity;
- TF-IDF relevance;
- response length and latency;
- heuristic hallucination-risk flags;
- category and difficulty slices;
- paired mean deltas, win rates, and 95% bootstrap confidence intervals;
- manual factuality, relevance, clarity, instruction-following, preference, and hallucination review.

Automated similarity and heuristic measures are diagnostics, not proof of factual correctness.

## Evaluation Results

Populate this section only after the reviewed experiment has completed.

| Metric | Base FLAN-T5 | LoRA model | Delta | 95% CI |
|---|---:|---:|---:|---|
| Instruction adherence | Not run | Not run | Not run | Not run |
| Response-quality rubric | Not run | Not run | Not run | Not run |
| BERTScore F1 | Not run | Not run | Not run | Not run |
| ROUGE-L F1 | Not run | Not run | Not run | Not run |
| Semantic similarity | Not run | Not run | Not run | Not run |
| Hallucination-risk flag rate | Not run | Not run | Not run | Not run |
| Average latency | Not run | Not run | Not run | Not run |

Source files after promotion:

- `outputs/base_model_metrics.json`
- `outputs/lora_model_metrics.json`
- `outputs/base_vs_lora_comparison.json`
- `outputs/per_example_base_vs_lora.csv`
- `outputs/release_manifest.json`

## Intended Use

- ML/Data Science educational explanations;
- portfolio demonstration of Transformer instruction tuning and PEFT;
- technical-learning and onboarding prototypes;
- study of small-model response evaluation;
- non-confidential examples related to quality analytics.

## Not Intended Use

- legal, medical, financial, immigration, safety-critical, or official advice;
- autonomous root-cause decisions;
- production use without independent testing and monitoring;
- processing private, proprietary, copyrighted, sensitive, or personally identifiable information;
- factual authority without human verification.

## Limitations

- The custom dataset is synthetic/self-authored rather than a production-scale audited curriculum.
- Teacher-generated candidates can contain errors despite validation and manual review.
- FLAN-T5-base has limited capacity for complex reasoning and code.
- Reference-similarity metrics do not establish factual correctness.
- Heuristic hallucination detection has false positives and false negatives.
- Results generalize only as far as the benchmark coverage supports.
- CPU deployment can have noticeable cold-start and generation latency.

## Bias and Risk Notes

The model can produce confident but incorrect explanations, incomplete caveats, misleading code, oversimplified comparisons, or recommendations influenced by dataset style. Review every generated answer before use and preserve human decision authority.

## Responsible Use

Educational and portfolio demonstration only. Do not submit sensitive content. Generated explanations and code require human review.

## Inference Example

```python
from src.config import ModelConfig
from src.inference_pipeline import InstructionAssistant

assistant = InstructionAssistant(
    ModelConfig(
        base_model_id="google/flan-t5-base",
        local_adapter_path="models/lora_adapter",
    )
)
result = assistant.generate(
    "Compare logistic regression and decision tree and include one limitation of each.",
    category="Algorithm comparison",
)
print(result["response"])
```

## Deployment

The Gradio Space loads the base model and a local or Hub-hosted PEFT adapter. Training never runs during app startup. Verify that inference metadata reports `lora_adapter`, not `base_model_fallback`, before sharing the demo.
