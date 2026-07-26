from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd

from src.inference_pipeline import summarize_text
from src.summarization_model import GenerationSettings

PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLE_PATH = PROJECT_ROOT / "data" / "sample_articles.csv"
GITHUB_URL = os.getenv(
    "GITHUB_URL", "https://github.com/<YOUR_GITHUB_USERNAME>/transformer-projects"
)
HF_MODEL_URL = os.getenv(
    "HF_MODEL_URL", "https://huggingface.co/<YOUR_USERNAME>/distilbart-summarization-portfolio"
)


def load_samples() -> pd.DataFrame:
    if not SAMPLE_PATH.exists():
        return pd.DataFrame(columns=["title", "article"])
    return pd.read_csv(SAMPLE_PATH)


SAMPLES = load_samples()
SAMPLE_MAP = dict(zip(SAMPLES.get("title", []), SAMPLES.get("article", [])))


def select_sample(title: str) -> str:
    return str(SAMPLE_MAP.get(title, ""))


def _settings(
    min_length: int,
    max_length: int,
    num_beams: int,
    length_penalty: float,
    no_repeat_ngram_size: int,
    early_stopping: bool,
) -> GenerationSettings:
    return GenerationSettings(
        min_length=int(min_length),
        max_length=int(max_length),
        num_beams=int(num_beams),
        length_penalty=float(length_penalty),
        no_repeat_ngram_size=int(no_repeat_ngram_size),
        early_stopping=bool(early_stopping),
    ).validate()


def _download_file(summary: str, details: dict[str, Any]) -> str:
    handle = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", prefix="transformer_summary_", delete=False, encoding="utf-8"
    )
    with handle:
        handle.write("GENERATED SUMMARY\n=================\n")
        handle.write(summary)
        handle.write("\n\nGENERATION DETAILS\n==================\n")
        handle.write(json.dumps(details, indent=2))
    return handle.name


def generate_summary(
    article: str,
    min_length: int,
    max_length: int,
    num_beams: int,
    length_penalty: float,
    no_repeat_ngram_size: int,
    early_stopping: bool,
) -> tuple[str, str, dict[str, Any], str | None, str]:
    try:
        settings = _settings(
            min_length,
            max_length,
            num_beams,
            length_penalty,
            no_repeat_ngram_size,
            early_stopping,
        )
        result = summarize_text(article, settings)
        details = result.to_dict()
        metrics = (
            f"### Generation metrics\n"
            f"- **Inference time:** {result.inference_seconds:.2f} seconds\n"
            f"- **Input length:** {result.input_words} words\n"
            f"- **Summary length:** {result.summary_words} words\n"
            f"- **Compression ratio:** {result.compression_ratio:.3f}\n"
            f"- **Chunks processed:** {result.chunks_processed}\n"
            f"- **Device:** {result.device}"
        )
        return result.summary, metrics, details, _download_file(result.summary, details), "✅ Summary generated successfully."
    except Exception as exc:
        return "", "", {"error": str(exc)}, None, f"❌ {exc}"


def compare_beam_settings(
    article: str,
    beam_choices: list[str],
    min_length: int,
    max_length: int,
    length_penalty: float,
    no_repeat_ngram_size: int,
) -> tuple[pd.DataFrame, str]:
    if not beam_choices:
        return pd.DataFrame(), "Select at least one beam count."
    rows: list[dict[str, Any]] = []
    try:
        for value in beam_choices:
            settings = _settings(
                min_length,
                max_length,
                int(value),
                length_penalty,
                no_repeat_ngram_size,
                True,
            )
            result = summarize_text(article, settings)
            rows.append(
                {
                    "num_beams": int(value),
                    "summary": result.summary,
                    "inference_seconds": round(result.inference_seconds, 3),
                    "summary_words": result.summary_words,
                    "compression_ratio": round(result.compression_ratio, 3),
                }
            )
        return pd.DataFrame(rows), "✅ Beam comparison completed."
    except Exception as exc:
        return pd.DataFrame(rows), f"❌ {exc}"


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Abstractive Text Summarization Transformer") as demo:
        gr.Markdown(
            """
# 📝 Abstractive Text Summarization Transformer

Generate concise summaries from news articles, quality reports, complaint narratives, and long text using a DistilBART encoder-decoder Transformer. Experiment with summary length and beam search, then inspect latency and compression metrics.

> ⚠️ **Responsible use:** Outputs may omit context, change facts, or hallucinate. Do not paste private, confidential, copyrighted, or personally identifiable text into a public demo. Do not rely on generated summaries for high-stakes decisions. Human review is required.
"""
        )

        with gr.Tabs():
            with gr.Tab("Generate Summary"):
                with gr.Row():
                    with gr.Column(scale=3):
                        sample_selector = gr.Dropdown(
                            choices=list(SAMPLE_MAP),
                            label="Load a safe sample article",
                            value=list(SAMPLE_MAP)[0] if SAMPLE_MAP else None,
                        )
                        load_sample_button = gr.Button("Load selected sample")
                        article = gr.Textbox(
                            label="Article or long text",
                            lines=18,
                            placeholder="Paste at least 25 words...",
                            value=next(iter(SAMPLE_MAP.values()), ""),
                        )
                    with gr.Column(scale=2):
                        gr.Markdown("### Summary controls")
                        min_length = gr.Slider(10, 100, value=30, step=5, label="Minimum summary length")
                        max_length = gr.Slider(40, 220, value=120, step=10, label="Maximum summary length")
                        num_beams = gr.Slider(1, 8, value=4, step=1, label="Number of beams")
                        length_penalty = gr.Slider(0.5, 3.0, value=2.0, step=0.1, label="Length penalty")
                        no_repeat = gr.Slider(0, 5, value=3, step=1, label="No-repeat n-gram size")
                        early_stopping = gr.Checkbox(value=True, label="Early stopping")
                        generate = gr.Button("Generate summary", variant="primary")
                        status = gr.Markdown()

                summary = gr.Textbox(label="Generated abstractive summary", lines=8)
                metrics = gr.Markdown()
                with gr.Accordion("Full generation details", open=False):
                    details = gr.JSON(label="Generation details")
                download = gr.File(label="Download summary and generation details")

                load_sample_button.click(select_sample, sample_selector, article)
                generate.click(
                    generate_summary,
                    inputs=[
                        article,
                        min_length,
                        max_length,
                        num_beams,
                        length_penalty,
                        no_repeat,
                        early_stopping,
                    ],
                    outputs=[summary, metrics, details, download, status],
                )

            with gr.Tab("Compare Beam Search"):
                gr.Markdown(
                    "Higher beam counts explore more candidate summaries but usually increase latency. Compare selected settings on the same source text."
                )
                beam_article = gr.Textbox(
                    label="Article",
                    lines=14,
                    value=next(iter(SAMPLE_MAP.values()), ""),
                )
                with gr.Row():
                    beam_choices = gr.CheckboxGroup(
                        choices=["1", "2", "4", "8"], value=["1", "4"], label="Beam counts"
                    )
                    beam_min = gr.Slider(10, 80, value=30, step=5, label="Minimum length")
                    beam_max = gr.Slider(40, 180, value=120, step=10, label="Maximum length")
                with gr.Row():
                    beam_penalty = gr.Slider(0.5, 3.0, value=2.0, step=0.1, label="Length penalty")
                    beam_no_repeat = gr.Slider(0, 5, value=3, step=1, label="No-repeat n-gram")
                compare_button = gr.Button("Run beam comparison", variant="primary")
                comparison_status = gr.Markdown()
                comparison_table = gr.Dataframe(label="Beam comparison", interactive=False, wrap=True)
                compare_button.click(
                    compare_beam_settings,
                    inputs=[
                        beam_article,
                        beam_choices,
                        beam_min,
                        beam_max,
                        beam_penalty,
                        beam_no_repeat,
                    ],
                    outputs=[comparison_table, comparison_status],
                )

            with gr.Tab("Transformer vs LSTM"):
                gr.Markdown(
                    """
## Honest comparison framework

This project does **not** invent metrics for the previous LSTM Seq2Seq model. Add actual LSTM predictions to `data/lstm_comparison_template.csv`, then run:

```bash
python scripts/compare_with_lstm.py --lstm-csv data/lstm_comparison_template.csv
```

| Dimension | LSTM Seq2Seq with Attention | DistilBART Transformer |
|---|---|---|
| Processing | Recurrent, step by step | Attention-based parallel encoding |
| Long-range context | Hidden-state bottleneck | Direct token-to-token attention |
| Pretraining | Project-specific | Large pretrained language model |
| Decoding controls | Custom implementation | Beam search, penalties, repetition controls |
| Published scores | Add only actual prior results | Run repository evaluation script |
"""
                )

            with gr.Tab("Model, Limits and Links"):
                gr.Markdown(
                    f"""
## Model details

- Base model: `sshleifer/distilbart-cnn-12-6`
- Task: English abstractive summarization
- Runtime: PyTorch, direct `AutoModelForSeq2SeqLM` generation
- Long-text strategy: token-aware chunks plus second-pass summary
- Training during startup: none

## Limitations

The default checkpoint is news-oriented. It may miss domain-specific quality language, alter numbers or entities, hallucinate, or over-compress. CPU inference can be slow, especially for long inputs and high beam counts. The system does not fact-check outputs.

## Links

- GitHub: {GITHUB_URL}
- Hugging Face model: {HF_MODEL_URL}
"""
                )

        gr.Markdown(
            "*Project 01 of the Transformer portfolio: ANN → RNN → LSTM → BiLSTM → CNN → Transformers.*"
        )
    return demo
