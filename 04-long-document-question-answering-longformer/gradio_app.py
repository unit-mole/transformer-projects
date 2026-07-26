from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd

from src.config import InferenceConfig, PROJECT_ROOT
from src.document_loader import (
    DocumentLoadingError,
    list_sample_documents,
    load_document,
    load_sample_document,
)
from src.inference_pipeline import InferenceValidationError, LongDocumentQAPipeline
from src.qa_model import ModelLoadError


CONFIG = InferenceConfig().validate()
PIPELINE = LongDocumentQAPipeline(CONFIG)
SAMPLE_NAMES = list_sample_documents(CONFIG.sample_directory)
QUESTIONS_PATH = PROJECT_ROOT / "data" / "sample_questions.csv"
METRICS_PATH = PROJECT_ROOT / "outputs" / "model_metrics.json"


CUSTOM_CSS = """
.gradio-container {max-width: 1320px !important;}
.evidence-box {
    border: 1px solid var(--border-color-primary);
    border-radius: 10px;
    padding: 14px;
    line-height: 1.6;
    background: var(--background-fill-secondary);
}
.evidence-box mark {
    padding: 2px 5px;
    border-radius: 4px;
    font-weight: 700;
}
.disclaimer {
    border-left: 5px solid #b45309;
    padding: 12px 16px;
    background: rgba(180, 83, 9, 0.08);
    border-radius: 8px;
}
"""


def _question_lookup() -> dict[str, str]:
    if not QUESTIONS_PATH.exists():
        return {}
    frame = pd.read_csv(QUESTIONS_PATH)
    if not {"document_name", "question"}.issubset(frame.columns):
        return {}
    return (
        frame.drop_duplicates("document_name")
        .set_index("document_name")["question"]
        .astype(str)
        .to_dict()
    )


QUESTION_LOOKUP = _question_lookup()


def load_sample_for_ui(sample_name: str) -> tuple[str, str, str]:
    if not sample_name:
        return "", "", "Select a sample document or upload your own file."
    try:
        document = load_sample_document(sample_name, CONFIG)
        question = QUESTION_LOOKUP.get(sample_name, "")
        status = (
            f"Loaded **{document.source_name}** — {document.word_count:,} words and "
            f"{document.character_count:,} characters."
        )
        return document.text, question, status
    except Exception as exc:
        return "", "", f"Unable to load sample: `{type(exc).__name__}: {exc}`"


def _format_confidence(value: float, label: str) -> str:
    return (
        f"{value:.6f} — {label}\n\n"
        "This is an uncalibrated model confidence proxy, not a probability that "
        "the answer is correct."
    )


def _diagnostic_payload(result: Any) -> dict[str, Any]:
    return {
        "source_name": result.source_name,
        "model_id": result.model_id,
        "document_characters": result.document_character_count,
        "document_words": result.document_word_count,
        "document_tokens": result.document_token_count,
        "runtime_window_tokens": result.requested_max_length,
        "model_supported_tokens": result.model_max_length,
        "overlapping_windows": result.window_count,
        "latency_seconds": round(result.latency_seconds, 4),
        "paragraph_index": result.paragraph_index,
        "answer_character_span": [
            result.answer_start_char,
            result.answer_end_char,
        ],
        "warnings": result.warnings,
        **result.diagnostics,
    }


def answer_document(
    uploaded_file: Any,
    sample_name: str,
    manual_text: str,
    question: str,
    max_length: int,
    stride: int,
) -> tuple[str, str, str, str, dict[str, Any], str]:
    try:
        document = load_document(
            uploaded_file=uploaded_file,
            manual_text=manual_text,
            sample_name=sample_name,
            config=CONFIG,
        )
        result = PIPELINE.answer(
            question=question,
            document_text=document.text,
            source_name=document.source_name,
            max_length=int(max_length),
            stride=int(stride),
        )
        warning_block = ""
        if result.warnings:
            warning_block = "\n\n**Review notes:**\n- " + "\n- ".join(result.warnings)
        status = (
            f"Completed QA over **{document.source_name}** in "
            f"**{result.latency_seconds:.2f} seconds** using "
            f"**{result.window_count} token window(s)**.{warning_block}"
        )
        return (
            result.answer,
            _format_confidence(result.confidence_proxy, result.confidence_label),
            result.supporting_paragraph
            or "No supporting paragraph could be identified.",
            result.highlighted_evidence_html,
            _diagnostic_payload(result),
            status,
        )
    except (
        DocumentLoadingError,
        InferenceValidationError,
        ModelLoadError,
        ValueError,
    ) as exc:
        message = f"{type(exc).__name__}: {exc}"
        return (
            "Unable to answer the question.",
            "0.000000 — no confidence proxy available",
            "",
            (
                "<div class='evidence-box'><strong>No evidence available.</strong>"
                f"<br>{message}</div>"
            ),
            {"error": message},
            f"**Request failed:** `{message}`",
        )
    except Exception as exc:  # defensive public-demo boundary
        message = f"{type(exc).__name__}: {exc}"
        return (
            "The application encountered an unexpected error.",
            "0.000000 — no confidence proxy available",
            "",
            "<div class='evidence-box'>No evidence available.</div>",
            {"error": message},
            "The request failed. Review the diagnostics and Space logs.",
        )


def load_saved_metrics() -> dict[str, Any]:
    if not METRICS_PATH.exists():
        return {"status": "not_run"}
    try:
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"status": "invalid_metrics_file"}


DEFAULT_SAMPLE = SAMPLE_NAMES[0] if SAMPLE_NAMES else None
DEFAULT_TEXT, DEFAULT_QUESTION, DEFAULT_STATUS = load_sample_for_ui(
    DEFAULT_SAMPLE or ""
)


with gr.Blocks(
    title="Long-Document Question Answering with Longformer",
) as demo:
    gr.Markdown(
        """
# 📄 Long-Document Question Answering with Longformer

Upload or select a document, ask a focused question, and inspect the predicted
answer, model confidence proxy, supporting paragraph, highlighted evidence, and
long-context diagnostics.

<div class="disclaimer">

**Responsible-use notice:** This educational portfolio demo may return
incorrect, incomplete, unsupported, or misleading answers. The confidence
value is only a model-based proxy. Do not upload private, confidential,
sensitive, copyrighted, proprietary, or personally identifiable documents.
Do not use the output as the sole basis for legal, medical, financial,
safety-critical, academic, official, or business-critical decisions.

</div>
        """
    )

    with gr.Tabs():
        with gr.Tab("Ask a Document"):
            with gr.Row():
                with gr.Column(scale=5):
                    uploaded_file = gr.File(
                        label="Upload document",
                        file_types=[".txt", ".md", ".csv", ".pdf"],
                        type="filepath",
                    )
                    sample_name = gr.Dropdown(
                        choices=SAMPLE_NAMES,
                        value=DEFAULT_SAMPLE,
                        label="Or choose a preloaded sample",
                    )
                    manual_text = gr.Textbox(
                        value=DEFAULT_TEXT,
                        label="Document text",
                        lines=16,
                        placeholder="Paste document text here.",
                    )
                with gr.Column(scale=4):
                    question = gr.Textbox(
                        value=DEFAULT_QUESTION,
                        label="Question",
                        lines=3,
                        placeholder="Ask a focused question answerable from the document.",
                    )
                    with gr.Row():
                        max_length = gr.Slider(
                            minimum=512,
                            maximum=4096,
                            value=CONFIG.max_length,
                            step=256,
                            label="Runtime context window (tokens)",
                        )
                        stride = gr.Slider(
                            minimum=0,
                            maximum=1024,
                            value=CONFIG.stride,
                            step=64,
                            label="Window overlap / stride",
                        )
                    ask_button = gr.Button("Ask Question", variant="primary")
                    input_status = gr.Markdown(DEFAULT_STATUS)

            gr.Markdown("## Model Output")
            with gr.Row():
                answer_output = gr.Textbox(label="Answer", lines=3)
                confidence_output = gr.Textbox(
                    label="Model confidence proxy",
                    lines=3,
                )
            supporting_output = gr.Textbox(
                label="Supporting paragraph",
                lines=7,
            )
            evidence_output = gr.HTML(label="Highlighted evidence")
            diagnostics_output = gr.JSON(label="Context and inference diagnostics")
            run_status = gr.Markdown()

            sample_name.change(
                fn=load_sample_for_ui,
                inputs=sample_name,
                outputs=[manual_text, question, input_status],
            )
            ask_button.click(
                fn=answer_document,
                inputs=[
                    uploaded_file,
                    sample_name,
                    manual_text,
                    question,
                    max_length,
                    stride,
                ],
                outputs=[
                    answer_output,
                    confidence_output,
                    supporting_output,
                    evidence_output,
                    diagnostics_output,
                    run_status,
                ],
            )

        with gr.Tab("Model and Evaluation"):
            gr.Markdown(
                f"""
## Selected checkpoint

`{CONFIG.model_id}`

The checkpoint is based on Longformer and is configured for extractive question
answering. Longformer combines local sliding-window attention with global
attention for selected tokens. This application uses overlapping token windows
when a document exceeds the selected runtime length and chooses the best valid
answer span across those windows.

The public CPU demo defaults to **{CONFIG.max_length} tokens per window** for
responsiveness. The underlying checkpoint supports up to approximately
**4,096 tokens** in one window.

## Evaluation

Run `python scripts/evaluate_model.py` to generate Exact Match, token-level F1,
evidence recall, latency, QA examples, and manual error analysis. Run
`python scripts/run_context_analysis.py` to create metrics by context-length
bucket. No metric is claimed until the evaluation scripts have produced it.

## Limitations

- The checkpoint was fine-tuned on SQuAD-style extractive QA, not every document domain.
- It cannot reliably answer questions whose answer is absent or only implied.
- Very long documents require multiple windows and may suffer boundary errors.
- PDF extraction works only for PDFs containing selectable text; OCR is excluded.
- Confidence values are not calibrated correctness probabilities.
                """
            )
            gr.JSON(value=load_saved_metrics(), label="Saved evaluation metrics")

        with gr.Tab("Portfolio Links"):
            gr.Markdown(
                """
- **GitHub repository:** `https://github.com/<YOUR_USERNAME>/transformer-models-projects`
- **Hugging Face Space:** `https://huggingface.co/spaces/<YOUR_USERNAME>/long-document-qa-longformer`
- **Base checkpoint:** `https://huggingface.co/valhalla/longformer-base-4096-finetuned-squadv1`

**One-line portfolio description:** Built a deployment-ready Longformer document
QA system that processes uploaded long documents and returns grounded answer
spans, a confidence proxy, supporting paragraphs, highlighted evidence, and
context-length evaluation.
                """
            )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(css=CUSTOM_CSS)
