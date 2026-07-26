"""Portfolio-ready Gradio application for Hugging Face Spaces."""

from __future__ import annotations

import json
import os
from pathlib import Path

import gradio as gr

from src.inference_pipeline import run_inference
from src.prompt_templates import PROMPT_CATEGORIES, all_examples

PROJECT_DIR = Path(__file__).resolve().parent
METRICS_PATH = PROJECT_DIR / "outputs" / "model_metrics.json"
MODEL_METADATA_PATH = PROJECT_DIR / "models" / "model_metadata.json"


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"status": "not_available"}


def generate_answer(category, instruction, supporting_input, max_new_tokens, temperature, top_p, repetition_penalty, model_choice):
    force_base = model_choice == "Base FLAN-T5 (comparison)"
    try:
        response, metadata = run_inference(
            instruction=instruction,
            input_text=supporting_input,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            force_base=force_base,
        )
        metadata["selected_category"] = category
        return response, metadata
    except Exception as exc:
        return (
            "The model could not be loaded. Confirm the BASE_MODEL_ID and ADAPTER_MODEL_ID settings, "
            "then review the Space logs. Training is intentionally not run when the app starts.",
            {"status": "error", "error_type": type(exc).__name__, "details": str(exc)},
        )


def load_category_example(category):
    prompts = PROMPT_CATEGORIES.get(category, [])
    return prompts[0] if prompts else ""


CSS = """
.gradio-container {max-width: 1180px !important; margin: auto;}
.disclaimer {border: 1px solid var(--border-color-primary); padding: 12px; border-radius: 10px;}
"""

with gr.Blocks(title="ML & Data Science Instruction-Tuned Assistant", fill_width=True) as demo:
    gr.Markdown(
        "# 🤖 ML & Data Science Instruction-Tuned Assistant\n"
        "A FLAN-T5 sequence-to-sequence model adapted with **LoRA / PEFT** on a curated ML and Data Science instruction dataset."
    )
    gr.Markdown(
        "**Responsible use:** This educational portfolio demo may produce incomplete, incorrect, outdated, biased, or hallucinated content. "
        "Do not use it for legal, medical, financial, immigration, safety-critical, or official decisions. "
        "Do not paste private, confidential, proprietary, copyrighted, or personally identifiable information.",
        elem_classes=["disclaimer"],
    )

    with gr.Tab("Assistant"):
        with gr.Row():
            with gr.Column(scale=3):
                category = gr.Dropdown(list(PROMPT_CATEGORIES), value="Concept explanation", label="Prompt category")
                instruction = gr.Textbox(
                    label="ML / Data Science question",
                    placeholder="Explain precision vs recall with a quality analytics example.",
                    lines=5,
                    max_lines=10,
                )
                supporting_input = gr.Textbox(
                    label="Optional supporting input",
                    placeholder="Add a scenario, data description, or constraints.",
                    lines=3,
                )
                model_choice = gr.Radio(
                    ["Domain LoRA adapter (preferred)", "Base FLAN-T5 (comparison)"],
                    value="Domain LoRA adapter (preferred)",
                    label="Model mode",
                )
                with gr.Accordion("Generation controls", open=False):
                    max_new_tokens = gr.Slider(32, 256, value=160, step=8, label="Maximum new tokens")
                    temperature = gr.Slider(0.0, 1.0, value=0.3, step=0.05, label="Temperature")
                    top_p = gr.Slider(0.5, 1.0, value=0.9, step=0.05, label="Top-p")
                    repetition_penalty = gr.Slider(1.0, 1.5, value=1.1, step=0.05, label="Repetition penalty")
                generate = gr.Button("Generate educational response", variant="primary")
                clear = gr.ClearButton([instruction, supporting_input])
            with gr.Column(scale=4):
                response = gr.Textbox(label="Assistant response", lines=16)
                metadata = gr.JSON(label="Inference metadata")
        gr.Examples(examples=all_examples(), inputs=[instruction], label="Example prompts")

    with gr.Tab("Model & LoRA"):
        gr.Markdown(
            "## Model design\n"
            "- **Base model:** `google/flan-t5-small`\n"
            "- **Task:** sequence-to-sequence instruction following\n"
            "- **Adaptation:** LoRA adapters through Hugging Face PEFT\n"
            "- **Deployment:** CPU-compatible Gradio Space; adapter loaded from the Hub\n\n"
            "LoRA trains small low-rank adapter weights while the base model remains frozen. "
            "The public app loads existing artifacts and never performs training at startup."
        )
        gr.JSON(value=_read_json(MODEL_METADATA_PATH), label="Model metadata")

    with gr.Tab("Evaluation"):
        gr.Markdown(
            "Evaluation includes instruction adherence, BERTScore, response relevance, latency, manual review, and hallucination analysis. "
            "Metrics are intentionally marked **not run** until the supplied evaluation scripts are executed against a trained adapter."
        )
        gr.JSON(value=_read_json(METRICS_PATH), label="Current evaluation status")

    with gr.Tab("Limitations"):
        gr.Markdown(
            "## Known limitations\n"
            "- The included dataset is a compact, self-authored portfolio starter and is not a substitute for a large expert-reviewed corpus.\n"
            "- Small models can omit caveats, generate incorrect code, or confuse related metrics.\n"
            "- Heuristic adherence, relevance, and hallucination flags require human interpretation.\n"
            "- The adapter repository must be configured after training. Without it, the app uses the base model for a transparent fallback.\n"
            "- CPU inference can be slower on the first request because model weights must be downloaded and loaded."
        )

    category.change(load_category_example, inputs=[category], outputs=[instruction])
    generate.click(
        generate_answer,
        inputs=[category, instruction, supporting_input, max_new_tokens, temperature, top_p, repetition_penalty, model_choice],
        outputs=[response, metadata],
    )


demo.queue(max_size=12)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")), css=CSS)
