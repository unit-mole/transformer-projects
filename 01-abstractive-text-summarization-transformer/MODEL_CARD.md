# Model Card: DistilBART Abstractive Summarization Portfolio System

## Model Details

- **System name:** Abstractive Text Summarization Transformer
- **Base model:** `sshleifer/distilbart-cnn-12-6`
- **Architecture:** Distilled BART encoder-decoder Transformer
- **Task:** English abstractive text summarization
- **Framework:** PyTorch and Hugging Face Transformers
- **License of this repository:** MIT
- **Base-model license:** Apache-2.0 according to the model repository; verify before redistribution or commercial use

## Dataset

The default app uses the pretrained model directly. Bundled demo data is original synthetic text created for this portfolio. Evaluation and optional fine-tuning scripts support bounded subsets of XSum and CNN/DailyMail loaded through Hugging Face Datasets. Full datasets are not redistributed in this repository.

## Intended Use

- Educational demonstration of Transformer summarization.
- Portfolio review by recruiters and technical reviewers.
- Experimentation with beam search and summary-length controls.
- Research prototypes for human-reviewed news, quality, complaint, case, root-cause, and report summarization.

## Not Intended For

- Autonomous legal, medical, financial, safety, compliance, academic, journalistic, or official decision-making.
- Processing confidential, personal, restricted, or proprietary text through a public demo.
- Producing authoritative factual records without source review.
- Plagiarism, misrepresentation, or bypassing copyright restrictions.

## Training Details

No fine-tuned weights are bundled. The default application loads the pretrained checkpoint. `scripts/train_model.py` provides an optional fine-tuning workflow using `Seq2SeqTrainer`; actual training settings and data provenance must be recorded before publishing a fine-tuned model.

## Generation Defaults

- Minimum length: 30 tokens
- Maximum length: 120 tokens
- Number of beams: 4
- Length penalty: 2.0
- No-repeat n-gram size: 3
- Early stopping: enabled
- Input chunk size: 900 tokens
- Chunk overlap: 64 tokens

## Evaluation

Supported metrics:

- ROUGE-1
- ROUGE-2
- ROUGE-L
- BERTScore
- Average/minimum/maximum inference time
- Compression ratio
- Generated/reference word counts

The repository intentionally ships `not_run` metric templates. Add actual results only after executing the evaluation script on a documented dataset and environment.

## Limitations and Risks

- News-domain pretraining may not transfer perfectly to manufacturing or quality text.
- The model may omit qualifiers, confuse entities, alter numbers/dates, or hallucinate details.
- Long-text map-reduce summarization can lose relationships across chunks.
- High beam counts increase CPU latency.
- Metric scores do not guarantee factual accuracy or usefulness.
- Dataset and pretrained-model biases can appear in summaries.

## Bias and Responsible Use

Review outputs for demographic, geographic, cultural, and domain bias. Do not expose private data. Preserve the original source for auditability. Require human review before operational use.

## Inference Example

```python
from src.inference_pipeline import summarize_text
from src.summarization_model import GenerationSettings

result = summarize_text(
    "Paste a sufficiently long article here.",
    GenerationSettings(num_beams=4, min_length=30, max_length=120),
)
print(result.summary)
print(result.to_dict())
```

## Deployment

The Gradio app is launched through `app.py`. The project is prepared for Hugging Face Spaces and Docker. No model training occurs during startup; the base checkpoint is downloaded from the Hugging Face Hub on first inference.
