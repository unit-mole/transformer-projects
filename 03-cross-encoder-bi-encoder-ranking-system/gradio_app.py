from __future__ import annotations

from dataclasses import asdict

import gradio as gr
import pandas as pd

from src.inference_pipeline import get_engine, run_search

DISPLAY_COLUMNS = [
    "retrieval_rank",
    "reranked_rank",
    "rank_movement",
    "document_id",
    "title",
    "category",
    "bi_encoder_score",
    "cross_encoder_score",
]


def _display_frame(frame: pd.DataFrame, reranked: bool) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=DISPLAY_COLUMNS)

    result = frame.copy()
    if not reranked:
        result["reranked_rank"] = pd.NA
        result["rank_movement"] = pd.NA
        result["cross_encoder_score"] = pd.NA

    for column in DISPLAY_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA

    result = result[DISPLAY_COLUMNS].copy()
    result["bi_encoder_score"] = pd.to_numeric(
        result["bi_encoder_score"], errors="coerce"
    ).round(4)
    result["cross_encoder_score"] = pd.to_numeric(
        result["cross_encoder_score"], errors="coerce"
    ).round(4)
    return result


def _tradeoff_note(use_reranker: bool, reranked: pd.DataFrame) -> str:
    if not use_reranker:
        return (
            "### Interpretation\n"
            "Only the bi-encoder stage was used. Documents were encoded independently, "
            "which supports fast retrieval and reusable document embeddings. "
            "No pairwise cross-encoder scoring was performed."
        )

    moved = 0
    if reranked is not None and not reranked.empty:
        moved = int(
            pd.to_numeric(
                reranked.get("rank_movement", pd.Series(dtype=float)),
                errors="coerce",
            )
            .fillna(0)
            .ne(0)
            .sum()
        )

    return (
        "### Interpretation\n"
        f"The cross-encoder reranked the selected candidates and changed the position "
        f"of **{moved}** result(s). Bi-encoder retrieval is fast because documents are "
        "encoded independently. Cross-encoder reranking is slower because it jointly "
        "scores each query-document pair, but it is applied only to the candidate set."
    )


def search_interface(
    query: str,
    sample_query: str,
    candidate_k: int,
    rerank_k: int,
    run_mode: str,
):
    final_query = (query or "").strip() or (sample_query or "").strip()
    if len(final_query) < 3:
        raise gr.Error("Enter a query or select a sample query.")

    use_reranker = run_mode == "Two-Stage Retrieval + Reranking"
    response = run_search(
        final_query,
        candidate_k=int(candidate_k),
        rerank_k=int(rerank_k),
        use_reranker=use_reranker,
    )

    bi_frame = _display_frame(response.candidates, reranked=False)
    reranked_frame = _display_frame(
        response.reranked_results,
        reranked=use_reranker,
    )

    latency = {
        "query": final_query,
        "mode": run_mode,
        "models": response.models,
        "index_preparation_ms": round(
            response.latency.index_preparation_ms, 2
        ),
        "query_embedding_ms": round(response.latency.query_embedding_ms, 2),
        "candidate_retrieval_ms": round(response.latency.retrieval_ms, 2),
        "cross_encoder_reranking_ms": round(
            response.latency.reranking_ms, 2
        ),
        "total_search_ms": round(response.latency.total_search_ms, 2),
    }
    return (
        final_query,
        bi_frame,
        reranked_frame,
        latency,
        _tradeoff_note(use_reranker, response.reranked_results),
    )


def build_demo() -> gr.Blocks:
    engine = get_engine()
    sample_queries = engine.sample_queries

    with gr.Blocks(
        title="DocRank360 — Two-Stage Transformer Search Ranking",
    ) as demo:
        gr.Markdown(
            """
            <div class="project-header">

            # 🔎 DocRank360
            ## Bi-Encoder Retrieval + Cross-Encoder Reranking

            A portfolio demonstration of a two-stage search-ranking system using
            **MiniLM Sentence-BERT** for fast candidate retrieval and an
            **MS MARCO MiniLM Cross-Encoder** for accurate reranking.

            </div>
            """
        )

        gr.Markdown(
            """
            <div class="notice">

            **Responsible-use notice:** This educational demo can return incomplete,
            biased, irrelevant, or misleading rankings. Scores are model-based
            relevance estimates, not guarantees of factual correctness or suitability.
            Do not upload private, confidential, copyrighted, proprietary, or personally
            identifiable text. Do not use this system as the sole basis for hiring,
            rejection, promotion, compensation, immigration, legal, or employment decisions.

            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                query = gr.Textbox(
                    label="Search query",
                    placeholder="Example: How can I find similar quality complaints?",
                    lines=2,
                )
                sample_query = gr.Dropdown(
                    choices=sample_queries,
                    value=sample_queries[0] if sample_queries else None,
                    label="Or choose a sample query",
                )
            with gr.Column(scale=1):
                candidate_k = gr.Slider(
                    minimum=3,
                    maximum=min(20, len(engine.dataset.documents)),
                    value=10,
                    step=1,
                    label="Bi-encoder candidate K",
                )
                rerank_k = gr.Slider(
                    minimum=1,
                    maximum=min(20, len(engine.dataset.documents)),
                    value=5,
                    step=1,
                    label="Cross-encoder rerank K",
                )
                run_mode = gr.Radio(
                    choices=[
                        "Bi-Encoder Retrieval Only",
                        "Two-Stage Retrieval + Reranking",
                    ],
                    value="Two-Stage Retrieval + Reranking",
                    label="Search mode",
                )

        search_button = gr.Button("Run Search", variant="primary")
        resolved_query = gr.Textbox(label="Executed query", interactive=False)

        with gr.Tabs():
            with gr.Tab("Stage 1 — Bi-Encoder Candidates"):
                bi_results = gr.Dataframe(
                    headers=DISPLAY_COLUMNS,
                    interactive=False,
                    wrap=True,
                    label="Candidate retrieval results",
                )
            with gr.Tab("Stage 2 — Cross-Encoder Reranking"):
                reranked_results = gr.Dataframe(
                    headers=DISPLAY_COLUMNS,
                    interactive=False,
                    wrap=True,
                    label="Reranked results",
                )
            with gr.Tab("Latency and Models"):
                latency_json = gr.JSON(label="Measured runtime details")

        explanation = gr.Markdown()

        with gr.Accordion("Architecture and evaluation", open=False):
            gr.Markdown(
                """
                **Pipeline**

                `User query → MiniLM query embedding → cosine candidate retrieval →`
                `MS MARCO MiniLM query-document scoring → final ranked results`

                **Required evaluation**

                - Recall@K for candidate coverage
                - MRR@10 for the first relevant result
                - nDCG@10 for graded ranking quality
                - MRR and nDCG improvement after reranking
                - Query encoding, retrieval, reranking, and total latency

                Run `python scripts/evaluate_model.py` to create actual project metrics.
                No evaluation values are hard-coded into this interface.
                """
            )

        with gr.Accordion("Model details and limitations", open=False):
            gr.Markdown(
                """
                - **Bi-encoder:** `sentence-transformers/all-MiniLM-L6-v2`
                - **Cross-encoder:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
                - **Index:** normalized NumPy cosine-similarity matrix for the small demo
                - **Data:** public-safe synthetic examples focused on quality analytics,
                  information retrieval, RAG, and job-search concepts
                - Cross-encoder scores are not calibrated probabilities.
                - Performance on this small demonstration corpus does not establish
                  production readiness or fairness.
                """
            )

        gr.Markdown(
            """
            **Portfolio links:**  
            GitHub: `https://github.com/<YOUR_GITHUB_USERNAME>/transformer-projects`  
            Hugging Face Space: `https://huggingface.co/spaces/<YOUR_HF_USERNAME>/<SPACE_NAME>`
            """
        )

        search_button.click(
            fn=search_interface,
            inputs=[
                query,
                sample_query,
                candidate_k,
                rerank_k,
                run_mode,
            ],
            outputs=[
                resolved_query,
                bi_results,
                reranked_results,
                latency_json,
                explanation,
            ],
        )

    return demo


if __name__ == "__main__":
    build_demo().launch()
