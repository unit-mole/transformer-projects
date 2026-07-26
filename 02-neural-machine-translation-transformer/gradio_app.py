from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd

from src.batch_translation import translate_csv
from src.config import load_model_metadata
from src.translation_pipeline import build_default_pipeline

PROJECT_ROOT = Path(__file__).resolve().parent
RESPONSIBLE_USE = """
> **Responsible use:** This educational demo can produce incorrect or incomplete
translations. The confidence value is a model-based proxy, not a guarantee.
Do not enter private, confidential, sensitive, legal, medical, financial,
immigration, safety-critical, or business-critical text. Human review is required.
"""


def translate_sentence(
    text: str,
    direction_label: str,
) -> tuple[str, str, str, float | None, float | None, dict[str, Any], str]:
    try:
        pipeline = build_default_pipeline()
        result = pipeline.translate(text, direction=direction_label)
        details = {
            "model_id": result.model_id,
            "device": result.device,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "confidence_method": result.confidence_method,
            "confidence_label": result.confidence_label,
            "warning": result.warning,
        }
        return (
            result.translated_text,
            result.detected_language.title(),
            result.direction_label,
            result.confidence_score,
            result.latency_seconds,
            details,
            "Translation completed. Review the result before use.",
        )
    except Exception as exc:
        return "", "", "", None, None, {"error": str(exc)}, f"Error: {exc}"


def preview_csv(file_path: str | None):
    if not file_path:
        return gr.update(choices=[], value=None), pd.DataFrame()
    try:
        dataframe = pd.read_csv(file_path)
    except Exception as exc:
        return gr.update(choices=[], value=None), pd.DataFrame(
            {"error": [f"Could not read CSV: {exc}"]}
        )
    columns = [str(column) for column in dataframe.columns]
    value = columns[0] if columns else None
    return gr.update(choices=columns, value=value), dataframe.head(10)


def translate_batch(
    file_path: str | None,
    text_column: str | None,
    direction_label: str,
    max_rows: int,
):
    if not file_path:
        return pd.DataFrame(), None, {}, "Upload a CSV file first."
    if not text_column:
        return pd.DataFrame(), None, {}, "Select the text column."

    try:
        output, download_path, summary = translate_csv(
            file_path,
            text_column=text_column,
            direction=direction_label,
            max_rows=int(max_rows),
        )
        return (
            output.head(100),
            download_path,
            summary,
            "Batch translation completed. Rows with mixed or uncertain text are "
            "reported as errors when Automatic direction is selected.",
        )
    except Exception as exc:
        return pd.DataFrame(), None, {"error": str(exc)}, f"Error: {exc}"


def build_demo() -> gr.Blocks:
    metadata = load_model_metadata()
    model_markdown = f"""
### Models

- English → Hindi: `{metadata['en_hi_model_id']}`
- Hindi → English: `{metadata['hi_en_model_id']}`
- Inference: lazy-loaded MarianMT encoder-decoder Transformers
- Metrics supported: SacreBLEU, chrF, latency, and manual error analysis
"""

    with gr.Blocks(title="English–Hindi Neural Machine Translation") as demo:
        gr.Markdown(
            """
# 🌐 English–Hindi Neural Machine Translation

Bidirectional MarianMT translation with automatic language detection,
sentence and CSV batch inference, confidence-proxy scoring, and latency tracking.
""",
        )
        gr.Markdown(RESPONSIBLE_USE)

        with gr.Tabs():
            with gr.Tab("Sentence Translation"):
                with gr.Row():
                    with gr.Column(scale=2):
                        source_text = gr.Textbox(
                            label="English or Hindi text",
                            lines=7,
                            placeholder=(
                                "Example: The quality report is ready.\n"
                                "उदाहरण: गुणवत्ता रिपोर्ट तैयार है।"
                            ),
                        )
                        direction = gr.Dropdown(
                            choices=[
                                "Automatic",
                                "English → Hindi",
                                "Hindi → English",
                            ],
                            value="Automatic",
                            label="Translation direction",
                        )
                        translate_button = gr.Button("Translate", variant="primary")
                        clear_button = gr.ClearButton(
                            [source_text, direction],
                            value="Clear",
                        )

                    with gr.Column(scale=2):
                        translated_text = gr.Textbox(
                            label="Translated text",
                            lines=7,
                            interactive=False,
                        )
                        with gr.Row():
                            detected_language = gr.Textbox(
                                label="Detected language",
                                interactive=False,
                            )
                            resolved_direction = gr.Textbox(
                                label="Resolved direction",
                                interactive=False,
                            )
                        with gr.Row():
                            confidence = gr.Number(
                                label="Confidence proxy",
                                precision=4,
                                interactive=False,
                            )
                            latency = gr.Number(
                                label="Latency (seconds)",
                                precision=4,
                                interactive=False,
                            )
                        details = gr.JSON(label="Inference details")
                        sentence_status = gr.Markdown()

                translate_button.click(
                    fn=translate_sentence,
                    inputs=[source_text, direction],
                    outputs=[
                        translated_text,
                        detected_language,
                        resolved_direction,
                        confidence,
                        latency,
                        details,
                        sentence_status,
                    ],
                )
                source_text.submit(
                    fn=translate_sentence,
                    inputs=[source_text, direction],
                    outputs=[
                        translated_text,
                        detected_language,
                        resolved_direction,
                        confidence,
                        latency,
                        details,
                        sentence_status,
                    ],
                )
                gr.Examples(
                    examples=[
                        ["The quality report is ready.", "Automatic"],
                        ["कृपया उत्पाद संख्या सत्यापित करें।", "Automatic"],
                        ["The sensor needs calibration.", "English → Hindi"],
                    ],
                    inputs=[source_text, direction],
                )

            with gr.Tab("Batch Translation"):
                gr.Markdown(
                    """
Upload a CSV, select the text column, and download a translated CSV.
Automatic mode records mixed or uncertain rows as errors instead of crashing.
"""
                )
                with gr.Row():
                    batch_file = gr.File(
                        label="CSV file",
                        file_types=[".csv"],
                        type="filepath",
                    )
                    text_column = gr.Dropdown(
                        choices=[],
                        label="Text column",
                        interactive=True,
                    )
                    batch_direction = gr.Dropdown(
                        choices=[
                            "Automatic",
                            "English → Hindi",
                            "Hindi → English",
                        ],
                        value="Automatic",
                        label="Direction",
                    )
                    max_rows = gr.Slider(
                        minimum=1,
                        maximum=500,
                        value=100,
                        step=1,
                        label="Maximum rows",
                    )
                input_preview = gr.Dataframe(
                    label="Input preview",
                    interactive=False,
                )
                batch_button = gr.Button("Translate CSV", variant="primary")
                batch_output = gr.Dataframe(
                    label="Translation results",
                    interactive=False,
                )
                batch_download = gr.File(label="Download translated CSV")
                batch_summary = gr.JSON(label="Batch summary")
                batch_status = gr.Markdown()

                batch_file.change(
                    fn=preview_csv,
                    inputs=batch_file,
                    outputs=[text_column, input_preview],
                )
                batch_button.click(
                    fn=translate_batch,
                    inputs=[
                        batch_file,
                        text_column,
                        batch_direction,
                        max_rows,
                    ],
                    outputs=[
                        batch_output,
                        batch_download,
                        batch_summary,
                        batch_status,
                    ],
                )

            with gr.Tab("Model, Evaluation, and Limitations"):
                gr.Markdown(model_markdown)
                gr.Markdown(
                    """
### Evaluation

Run `python scripts/evaluate_model.py` to generate direction-wise SacreBLEU,
chrF, latency summaries, and example translations. Metrics are not shown here
until they have been produced by an actual run.

### Manual error analysis

Review named entities, numbers, gender, tense, honorifics, word order,
under-translation, over-translation, mixed-language text, and domain terms.

### Limitations

- Automatic metrics do not guarantee semantic correctness.
- Domain-specific terminology can be mistranslated.
- Cold-start downloads and CPU inference can add latency.
- The confidence proxy is not calibrated.
- Public demos must not receive private or sensitive text.

### Portfolio links

- GitHub: `<YOUR_GITHUB_REPOSITORY_URL>`
- Hugging Face model: `<YOUR_HUGGINGFACE_MODEL_URL>`
"""
                )

    return demo
