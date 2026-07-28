# Portfolio-Grade Fine-Tuning and Evaluation Workflow

This workflow turns Project 02 into a complete experiment rather than only a pretrained-model application.

## Final experiment design

```text
IIT Bombay English-Hindi Parallel Corpus
                 │
                 ├── Deterministic training subset
                 ├── Official validation split
                 └── Held-out test subset
                              │
              ┌───────────────┴───────────────┐
              │                               │
      Pretrained MarianMT             Fine-tuned MarianMT
       EN→HI and HI→EN                 EN→HI and HI→EN
              │                               │
              └───────────────┬───────────────┘
                              │
        SacreBLEU + signature, chrF, chrF++, TER
        latency, throughput, GPU memory, diagnostics
        bootstrap intervals and paired comparison
                              │
                 Manual review of weak examples
                              │
      JSON + CSV + PNG outputs + Static Space metrics
```

## 1. Install the evaluation environment

Use your existing CUDA-enabled PyTorch environment. Confirm that the GPU is visible:

```cmd
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Then install the additional packages:

```cmd
pip install -r requirements-evaluation.txt
```

## 2. Open the notebook

```cmd
jupyter lab notebooks/03_portfolio_grade_marianmt_finetuning_evaluation.ipynb
```

Run the `quick` profile first. When it succeeds, change the profile to `portfolio` and rerun from the beginning.

## 3. Recommended final portfolio profile

- training pairs: 50,000
- validation pairs: 520
- held-out test pairs: 1,000
- epochs: 2
- both directions: EN→HI and HI→EN
- bootstrap resamples: 500
- human-reviewed examples: 30

The `full` profile uses 100,000 training pairs and all 2,507 official test pairs, but it is optional.

## 4. Outputs created automatically

```text
outputs/portfolio_evaluation/
├── dataset_manifest.json
├── pretrained/
│   ├── metrics_en_hi.json
│   ├── metrics_hi_en.json
│   └── metrics_combined.json
├── fine_tuned/
│   ├── metrics_en_hi.json
│   ├── metrics_hi_en.json
│   └── metrics_combined.json
├── training/
│   ├── en_hi/training_history.json
│   ├── en_hi/training_summary.json
│   ├── hi_en/training_history.json
│   └── hi_en/training_summary.json
├── model_comparison.csv
├── comparison_summary.json
├── manual_error_analysis_candidates.csv
├── manual_error_analysis_summary.json
└── plots/
    ├── comparison_sacrebleu.png
    ├── comparison_chrf.png
    ├── comparison_ter.png
    └── comparison_average_latency_seconds.png
```

The workflow also updates the existing recruiter-facing files:

```text
outputs/sacrebleu_scores.json
outputs/chrf_scores.json
outputs/translation_latency_results.json
outputs/model_metrics.json
outputs/model_comparison.csv
web/data/evaluation-results.json
```

## 5. Manual error analysis

Open `outputs/portfolio_evaluation/manual_error_analysis_candidates.csv` in Excel. Review at least 30 examples and fill:

- `human_error_category`
- `human_severity`
- `human_translation_quality`
- `human_notes`

Suggested categories:

```text
good_translation
word_order
missing_information
named_entity
number_or_date
gender_or_tense
over_literal
under_translation
over_translation
mixed_language
technical_terminology
other
```

Save the CSV and rerun the notebook's manual-summary and sync cells.

## 6. Optional command-line execution

The notebook is recommended because it displays every stage. The same stages can be run from CMD:

```cmd
python scripts/run_portfolio_evaluation.py --stage prepare --profile portfolio
python scripts/run_portfolio_evaluation.py --stage pretrained --profile portfolio
python scripts/run_portfolio_evaluation.py --stage finetune --profile portfolio
python scripts/run_portfolio_evaluation.py --stage fine-tuned --profile portfolio
python scripts/run_portfolio_evaluation.py --stage compare --profile portfolio
python scripts/run_portfolio_evaluation.py --stage manual-template --profile portfolio
```

After completing the manual CSV:

```cmd
python scripts/run_portfolio_evaluation.py --stage manual-summary --profile portfolio
python scripts/run_portfolio_evaluation.py --stage sync --profile portfolio
```

## 7. What to commit

Commit:

- notebook and source code;
- configuration and documentation;
- aggregate JSON metrics;
- model-comparison CSV;
- plots;
- completed 30-row manual error analysis;
- Static Space evaluation JSON.

Do not commit:

- fine-tuned model weights;
- trainer checkpoints;
- full train/validation/test CSV files;
- Hugging Face cache files;
- full 1,000-row prediction exports.

The model weights will later be uploaded to dedicated Hugging Face Model repositories. The Static Space currently uses pretrained quantized ONNX models; a later ONNX conversion step is required before claiming that browser inference uses the fine-tuned models.
