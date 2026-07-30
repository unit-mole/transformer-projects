"""Portfolio-ready Gradio interface for the ML/Data Science learning assistant."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import gradio as gr

from src.config import ModelConfig, PROJECT_ROOT
from src.data_preprocessing import load_jsonl
from src.inference_pipeline import InstructionAssistant

PROJECT_TITLE = "ML & Data Science Instruction-Tuned Assistant"
GITHUB_URL = "https://github.com/<your-github-username>/transformer-models-projects"
MODEL_URL = "https://huggingface.co/<your-huggingface-username>/<adapter-repository>"
RESPONSIBLE_USE = (
    "**Responsible use:** Educational portfolio demo only. Responses may be incomplete, outdated, biased, "
    "or incorrect. Do not use this assistant for legal, medical, financial, immigration, safety-critical, "
    "or official decisions. Do not paste private, confidential, proprietary, copyrighted, or personally "
    "identifiable information. Human review is required."
)

CATEGORIES = [
    "Concept explanation",
    "Algorithm comparison",
    "Metric explanation",
    "Example generation",
    "Beginner-friendly explanation",
    "Interview-style answer",
    "Small code example",
    "Data Science workflow",
    "ML project guidance",
    "Quality analytics",
]

ASSISTANT = InstructionAssistant()


def _load_samples() -> List[List[str]]:
    path = PROJECT_ROOT / "data" / "sample_instructions.jsonl"
    try:
        records = load_jsonl(path)
        return [[str(r["instruction"]), str(r["category"]), str(r.get("input", ""))] for r in records]
    except Exception:
        return [
            ["Explain random forest in simple terms.", "Concept explanation", ""],
            ["Compare logistic regression and decision tree.", "Algorithm comparison", ""],
            ["Explain precision versus recall with a quality analytics example.", "Quality analytics", ""],
        ]


def _load_metrics() -> Dict[str, object]:
    lora_path = PROJECT_ROOT / "outputs" / "lora_model_metrics.json"
    comparison_path = PROJECT_ROOT / "outputs" / "base_vs_lora_comparison.json"
    readiness_path = PROJECT_ROOT / "outputs" / "portfolio_readiness_report.json"
    if not lora_path.exists() or not comparison_path.exists():
        return {
            "status": "not_run",
            "message": "Run notebooks/05_full_training_evaluation_pipeline.ipynb, complete human review, and promote the experiment.",
            "expected_metrics": [
                "instruction adherence",
                "response-quality rubric",
                "BERTScore F1",
                "ROUGE-L F1",
                "semantic similarity",
                "hallucination-risk flag rate",
                "latency",
                "base-versus-LoRA confidence intervals",
            ],
        }
    result: Dict[str, object] = {
        "lora_model": json.loads(lora_path.read_text(encoding="utf-8")),
        "base_vs_lora": json.loads(comparison_path.read_text(encoding="utf-8")),
    }
    if readiness_path.exists():
        result["portfolio_readiness"] = json.loads(readiness_path.read_text(encoding="utf-8"))
    return result


def _model_summary() -> str:
    cfg = ModelConfig()
    adapter = cfg.adapter_id or cfg.local_adapter_path
    return (
        f"**Base model:** `{cfg.base_model_id}`  \n"
        f"**Adapter source:** `{adapter}`  \n"
        "**Method:** LoRA / PEFT for sequence-to-sequence instruction tuning  \n"
        "**Deployment behavior:** the app loads model artifacts for inference only; it never trains during startup."
    )


def respond(
    prompt: str,
    category: str,
    context: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
):
    try:
        result = ASSISTANT.generate(
            instruction=prompt,
            category=category,
            input_text=context,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            repetition_penalty=float(repetition_penalty),
        )
        status = (
            f"**Mode:** `{result.get('model_mode', 'unknown')}` · "
            f"**Latency:** `{result.get('latency_seconds', 0.0):.2f}s`"
        )
        metadata = {k: v for k, v in result.items() if k != "response"}
        return result["response"], status, metadata
    except Exception as exc:
        message = (
            "The model could not be loaded or used. Confirm internet access, package installation, "
            "and the adapter path/Hub ID. Technical detail: " + str(exc)
        )
        return message, "**Status:** generation error", {"error": str(exc)}


APP_THEME = gr.themes.Soft()
APP_CSS = """
    .hero {text-align:center; max-width:1000px; margin:0 auto 1rem auto;}
    .disclaimer {border-left:4px solid #d97706; padding:0.8rem 1rem; background:rgba(217,119,6,0.08);}
    .footer {text-align:center; opacity:0.8; font-size:0.9rem;}
    """


def build_demo() -> gr.Blocks:
    samples = _load_samples()
    metrics = _load_metrics()
    with gr.Blocks(title=PROJECT_TITLE) as demo:
        gr.Markdown(
            f"""
            <div class="hero">
            <h1>🤖 {PROJECT_TITLE}</h1>
            <p>Project 05 — FLAN-T5 instruction tuning with LoRA/PEFT for educational Machine Learning and Data Science support.</p>
            </div>
            """
        )
        gr.Markdown(RESPONSIBLE_USE, elem_classes=["disclaimer"])

        with gr.Row():
            with gr.Column(scale=3):
                category = gr.Dropdown(CATEGORIES, value=CATEGORIES[0], label="Prompt category")
                prompt = gr.Textbox(
                    label="Your ML/Data Science question",
                    placeholder="Example: Compare random forest and gradient boosting.",
                    lines=4,
                )
                context = gr.Textbox(
                    label="Optional context or constraints",
                    placeholder="Example: Explain for a beginner and include a quality analytics example.",
                    lines=3,
                )
                generate = gr.Button("Generate educational response", variant="primary")
            with gr.Column(scale=2):
                with gr.Accordion("Generation controls", open=True):
                    max_new_tokens = gr.Slider(48, 320, value=220, step=8, label="Maximum new tokens")
                    temperature = gr.Slider(0.0, 1.0, value=0.2, step=0.05, label="Temperature")
                    top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")
                    repetition_penalty = gr.Slider(1.0, 1.5, value=1.12, step=0.01, label="Repetition penalty")

        response = gr.Textbox(label="Assistant response", lines=12, buttons=["copy"])
        status = gr.Markdown("**Mode:** model loads on the first request")
        metadata = gr.JSON(label="Inference metadata", open=False)

        gr.Examples(
            examples=samples,
            inputs=[prompt, category, context],
            label="Sample prompts by capability",
        )

        with gr.Tabs():
            with gr.Tab("Model details"):
                gr.Markdown(_model_summary())
                gr.Markdown(
                    "LoRA freezes the base model and learns small low-rank adapter weights in selected attention modules. "
                    "The adapter can be stored locally or loaded from a Hugging Face model repository."
                )
            with gr.Tab("Evaluation"):
                gr.JSON(value=metrics, label="Recorded evaluation results")
                gr.Markdown(
                    "Metrics are populated only after the full training/evaluation notebook is run and reviewed. BERTScore is calculated only "
                    "for prompts with reference answers and is not treated as a factuality metric."
                )
            with gr.Tab("Hallucination analysis"):
                gr.Markdown(
                    "The project combines heuristic flags with manual review. Reviewers check false definitions, "
                    "unsupported claims, incorrect comparisons, code errors, missing caveats, and overconfident language."
                )
            with gr.Tab("Limitations"):
                gr.Markdown(
                    "FLAN-T5-base still has limited reasoning depth and knowledge. Output quality depends on the "
                    "instruction dataset and adapter. The included evaluation heuristics are diagnostic tools, not proof "
                    "of correctness. Public CPU Spaces can have slower first-request latency."
                )

        gr.Markdown(
            f"<div class='footer'>GitHub: {GITHUB_URL} · Model/adapter: {MODEL_URL}</div>"
        )

        generate.click(
            fn=respond,
            inputs=[prompt, category, context, max_new_tokens, temperature, top_p, repetition_penalty],
            outputs=[response, status, metadata],
            api_name="generate_ml_ds_response",
        )
        prompt.submit(
            fn=respond,
            inputs=[prompt, category, context, max_new_tokens, temperature, top_p, repetition_penalty],
            outputs=[response, status, metadata],
            api_name=False,
        )

    return demo


demo = build_demo()
