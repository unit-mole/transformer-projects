# Model Card — English–Hindi MarianMT Translation System

## Model details

| Field | Value |
|---|---|
| Project | `02-neural-machine-translation-transformer` |
| Task | Neural machine translation |
| Architecture | Marian encoder-decoder Transformer |
| English→Hindi base model | `Helsinki-NLP/opus-mt-en-hi` |
| Hindi→English base model | `Helsinki-NLP/opus-mt-hi-en` |
| Languages | English and Hindi |
| Default inference mode | Pretrained directional models |
| Framework | PyTorch + Hugging Face Transformers |
| Deployment | Gradio / Hugging Face Spaces |

## Intended use

Educational demonstrations, portfolio review, translation experimentation, model comparison, and low-risk prototyping with human review.

## Not intended for

Legal, medical, financial, immigration, safety-critical, academic-integrity, official, or business-critical decisions; confidential data; fully automated publication; or use without human verification.

## Dataset support

The preprocessing and optional training code supports:

- `cfilt/iitb-english-hindi`;
- permitted English–Hindi parallel CSV files;
- the safe sample pairs included in `data/`.

The default application uses pretrained models and does not claim that the included sample data was used to train them.

## Preprocessing

- Unicode NFKC normalization;
- HTML entity decoding and tag removal;
- control-character cleanup;
- whitespace normalization;
- preservation of Devanagari, punctuation, numbers, named entities, and case where useful;
- configurable length limits;
- duplicate and missing-pair removal for training data.

## Generation

- 4-beam search;
- early stopping;
- maximum source length of 512 tokens;
- maximum 256 new target tokens;
- 3-gram repetition blocking;
- renormalized logits.

## Confidence proxy

The displayed value is derived from generation sequence scores when available. It is bounded to `[0, 1]`, but is not calibrated to real-world correctness. A fallback heuristic uses length ratio, repetition, and unknown-token behavior. The method is returned with every result.

## Evaluation

Supported metrics:

- SacreBLEU;
- chrF;
- per-sentence latency;
- direction-wise latency summaries;
- example-level outputs;
- manual error analysis.

No project-specific scores are claimed until `scripts/evaluate_model.py` is run. Committed metric files intentionally contain `null`.

## Known risks and limitations

- word-order, gender, tense, honorific, named-entity, and number errors;
- over-literal or incomplete translation;
- domain terminology failures;
- mixed-script uncertainty;
- performance variation by sentence length and hardware;
- biases inherited from training data;
- confidence miscalibration.

## Example inference

```python
from src.translation_pipeline import build_default_pipeline

pipeline = build_default_pipeline()
result = pipeline.translate("The quality report is ready.", direction="en_hi")
print(result.translated_text)
```

## Deployment

The application loads model weights directly from the Hugging Face Hub. Fine-tuned checkpoints should be pushed to a separate model repository and referenced through environment variables or `models/model_metadata.json`.
