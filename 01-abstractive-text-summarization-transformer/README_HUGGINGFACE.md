# Hugging Face Space Card — Abstractive Text Summarization Transformer

## Description

Generate concise abstractive summaries from news articles and long text using DistilBART. Adjust minimum/maximum summary length, beam count, length penalty, repetition control, and early stopping. The app reports latency, compression ratio, input/output lengths, chunks processed, model, and device.

## How to Use

1. Paste an article or select a bundled sample.
2. Adjust generation controls.
3. Select **Generate summary**.
4. Review the output and metrics.
5. Download the generated summary if needed.
6. Use the beam-comparison tab to inspect speed-quality trade-offs.

## Model

- Base model: `sshleifer/distilbart-cnn-12-6`
- Task: English abstractive summarization
- Architecture: distilled BART encoder-decoder Transformer
- Inference: direct `AutoTokenizer` + `AutoModelForSeq2SeqLM`
- Training at startup: none

## Inputs and Outputs

- Input: English article or long-form text.
- Output: Generated summary plus latency, compression ratio, word counts, chunks processed, and runtime information.

## Evaluation

Repository scripts support ROUGE-1, ROUGE-2, ROUGE-L, BERTScore, compression ratio, and inference-time reporting. Published metrics must come from an actual evaluation run; no result is fabricated in this template.

## Transformer vs LSTM

The repository contains a comparison framework for actual outputs from the prior LSTM Seq2Seq summarization project. The demo explains architectural differences, while offline scripts calculate comparable metrics once real LSTM predictions are supplied.

## Responsible Use

Generated summaries can omit context, distort facts, or hallucinate. Do not paste confidential or sensitive text into a public Space. Do not rely on outputs for high-stakes decisions. Human review is required.

## Links

- GitHub: `https://github.com/<YOUR_GITHUB_USERNAME>/transformer-projects`
- Model repository: `https://huggingface.co/<YOUR_USERNAME>/distilbart-summarization-portfolio`

## Portfolio Note

This is Project 01 in a ten-project Transformer portfolio progressing from sequence-to-sequence generation to retrieval, multimodal AI, instruction tuning, and RAG.
