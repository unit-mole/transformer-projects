from __future__ import annotations

import json
import platform
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch

from .artifacts import (
    make_portfolio_markdown,
    plot_latency,
    plot_metric_comparison,
    plot_recall_curves,
    plot_reranking_delta,
    update_latest_alias,
    write_json,
)
from .baselines import BM25Retriever, TfidfRetriever
from .beir_loader import BEIRDataset, load_beir_dataset
from .metrics import evaluate_rankings, paired_bootstrap_delta
from .transformer_retrieval import (
    DenseTransformerRetriever,
    TransformerCrossEncoderReranker,
    resolve_device,
)


@dataclass
class BenchmarkConfig:
    project_root: Path
    datasets: tuple[str, ...] = ("scifact", "nfcorpus")
    split: str = "test"
    bi_encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    model_label: str = "base_minilm"
    device: str = "auto"
    candidate_k: int = 100
    rerank_k: int = 100
    recall_ks: tuple[int, ...] = (1, 3, 5, 10, 20, 50, 100)
    bi_encoder_batch_size: int = 128
    cross_encoder_batch_size: int = 64
    max_queries: int | None = None
    bootstrap_samples: int = 2_000
    random_seed: int = 42
    verify_md5: bool = True
    run_tfidf: bool = True
    run_bm25: bool = True
    run_cross_encoder: bool = True
    run_name: str | None = None

    @property
    def benchmark_cache_dir(self) -> Path:
        return self.project_root / "data" / "benchmarks"

    @property
    def benchmark_output_root(self) -> Path:
        return self.project_root / "outputs" / "benchmark"


@dataclass(frozen=True)
class BenchmarkRunResult:
    run_directory: Path
    latest_directory: Path
    summary: pd.DataFrame
    latency: pd.DataFrame
    per_query: pd.DataFrame
    bootstrap: dict[str, Any]
    metadata: list[dict[str, Any]]


def hardware_metadata(device: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "requested_device": device,
        "resolved_device": resolve_device(device),
    }
    if torch.cuda.is_available():
        payload.update(
            {
                "cuda_version": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_count": torch.cuda.device_count(),
                "gpu_total_memory_gb": round(
                    torch.cuda.get_device_properties(0).total_memory / 1024**3, 3
                ),
                "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            }
        )
    return payload


def _summary_row(
    *,
    dataset: str,
    approach: str,
    metrics: dict[str, float],
    index_build_ms: float,
    mean_query_ms: float,
    model_label: str,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "approach": approach,
        "model_label": model_label,
        **metrics,
        "index_build_ms": float(index_build_ms),
        "mean_query_ms": float(mean_query_ms),
    }


def _evaluate(
    dataset: BEIRDataset,
    approach: str,
    rankings: dict[str, list[str]],
    recall_ks: tuple[int, ...],
) -> tuple[pd.DataFrame, dict[str, float]]:
    details, summary = evaluate_rankings(
        rankings,
        dataset.qrels,
        recall_ks=recall_ks,
        mrr_k=10,
        ndcg_k=10,
        map_k=100,
    )
    details.insert(0, "approach", approach)
    details.insert(0, "dataset", dataset.name)
    return details, summary


def _ranking_examples(
    dataset: BEIRDataset,
    dense_rankings: dict[str, list[str]],
    reranked_rankings: dict[str, list[str]],
    dense_details: pd.DataFrame,
    reranked_details: pd.DataFrame,
) -> pd.DataFrame:
    before = dense_details.set_index("query_id")
    after = reranked_details.set_index("query_id")
    rows: list[dict[str, Any]] = []
    for query_id in dataset.query_ids:
        before_ndcg = float(before.loc[query_id, "ndcg_at_10"])
        after_ndcg = float(after.loc[query_id, "ndcg_at_10"])
        dense_top = dense_rankings.get(query_id, [""])[0]
        reranked_top = reranked_rankings.get(query_id, [""])[0]
        rows.append(
            {
                "dataset": dataset.name,
                "query_id": query_id,
                "query": dataset.queries[query_id],
                "bi_encoder_top_document_id": dense_top,
                "bi_encoder_top_title": dataset.corpus.get(dense_top, {}).get("title", ""),
                "reranked_top_document_id": reranked_top,
                "reranked_top_title": dataset.corpus.get(reranked_top, {}).get("title", ""),
                "bi_encoder_ndcg_at_10": before_ndcg,
                "reranked_ndcg_at_10": after_ndcg,
                "ndcg_delta": after_ndcg - before_ndcg,
                "top_result_changed": dense_top != reranked_top,
            }
        )
    frame = pd.DataFrame(rows)
    return frame.sort_values("ndcg_delta", ascending=False).reset_index(drop=True)


def run_benchmark_suite(config: BenchmarkConfig) -> BenchmarkRunResult:
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.random_seed)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = config.run_name or f"{config.model_label}-{timestamp}"
    run_dir = config.benchmark_output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    write_json(run_dir / "benchmark_config.json", {
        **asdict(config),
        "project_root": str(config.project_root),
    })
    write_json(run_dir / "hardware.json", hardware_metadata(config.device))

    summary_rows: list[dict[str, Any]] = []
    latency_rows: list[dict[str, Any]] = []
    all_per_query: list[pd.DataFrame] = []
    metadata: list[dict[str, Any]] = []
    bootstrap_results: dict[str, Any] = {}
    all_examples: list[pd.DataFrame] = []

    for dataset_name in config.datasets:
        print(f"\n{'=' * 80}\nBenchmarking {dataset_name}\n{'=' * 80}")
        dataset = load_beir_dataset(
            dataset_name,
            config.benchmark_cache_dir,
            split=config.split,
            max_queries=config.max_queries,
            verify_md5=config.verify_md5,
        )
        metadata.append(dataset.metadata())
        document_ids = dataset.corpus_ids
        documents = dataset.corpus_texts
        query_ids = dataset.query_ids
        queries = dataset.query_texts

        if config.run_tfidf:
            tfidf = TfidfRetriever()
            index_ms = tfidf.fit(documents)
            output = tfidf.search(
                query_ids, queries, document_ids, top_k=config.candidate_k
            )
            details, metrics = _evaluate(
                dataset, "tfidf", output.rankings, config.recall_ks
            )
            all_per_query.append(details)
            summary_rows.append(
                _summary_row(
                    dataset=dataset.name,
                    approach="tfidf",
                    metrics=metrics,
                    index_build_ms=index_ms,
                    mean_query_ms=output.mean_query_ms,
                    model_label=config.model_label,
                )
            )
            latency_rows.append(
                {
                    "dataset": dataset.name,
                    "approach": "tfidf",
                    "index_build_ms": index_ms,
                    "total_query_ms": output.total_query_ms,
                    "mean_query_ms": output.mean_query_ms,
                }
            )

        if config.run_bm25:
            bm25 = BM25Retriever()
            index_ms = bm25.fit(documents)
            output = bm25.search(
                query_ids, queries, document_ids, top_k=config.candidate_k
            )
            details, metrics = _evaluate(
                dataset, "bm25", output.rankings, config.recall_ks
            )
            all_per_query.append(details)
            summary_rows.append(
                _summary_row(
                    dataset=dataset.name,
                    approach="bm25",
                    metrics=metrics,
                    index_build_ms=index_ms,
                    mean_query_ms=output.mean_query_ms,
                    model_label=config.model_label,
                )
            )
            latency_rows.append(
                {
                    "dataset": dataset.name,
                    "approach": "bm25",
                    "index_build_ms": index_ms,
                    "total_query_ms": output.total_query_ms,
                    "mean_query_ms": output.mean_query_ms,
                }
            )

        dense = DenseTransformerRetriever(
            config.bi_encoder_model,
            device=config.device,
            batch_size=config.bi_encoder_batch_size,
        )
        dense_output = dense.retrieve(
            query_ids,
            queries,
            document_ids,
            documents,
            top_k=config.candidate_k,
        )
        dense_details, dense_metrics = _evaluate(
            dataset, "bi_encoder", dense_output.rankings, config.recall_ks
        )
        all_per_query.append(dense_details)
        summary_rows.append(
            _summary_row(
                dataset=dataset.name,
                approach="bi_encoder",
                metrics=dense_metrics,
                index_build_ms=dense_output.corpus_embedding_ms,
                mean_query_ms=dense_output.mean_query_ms,
                model_label=config.model_label,
            )
        )
        latency_rows.append(
            {
                "dataset": dataset.name,
                "approach": "bi_encoder",
                "index_build_ms": dense_output.corpus_embedding_ms,
                "query_embedding_ms": dense_output.query_embedding_ms,
                "search_ms": dense_output.search_ms,
                "mean_query_ms": dense_output.mean_query_ms,
                "device": dense_output.device,
            }
        )

        if config.run_cross_encoder:
            reranker = TransformerCrossEncoderReranker(
                config.cross_encoder_model,
                device=config.device,
                batch_size=config.cross_encoder_batch_size,
            )
            reranked_output = reranker.rerank(
                query_ids,
                dataset.queries,
                dense_output.rankings,
                dataset.corpus,
                rerank_k=min(config.rerank_k, config.candidate_k),
            )
            reranked_details, reranked_metrics = _evaluate(
                dataset,
                "bi_encoder_plus_cross_encoder",
                reranked_output.rankings,
                config.recall_ks,
            )
            all_per_query.append(reranked_details)
            summary_rows.append(
                _summary_row(
                    dataset=dataset.name,
                    approach="bi_encoder_plus_cross_encoder",
                    metrics=reranked_metrics,
                    index_build_ms=dense_output.corpus_embedding_ms,
                    mean_query_ms=(
                        dense_output.mean_query_ms + reranked_output.mean_query_ms
                    ),
                    model_label=config.model_label,
                )
            )
            latency_rows.append(
                {
                    "dataset": dataset.name,
                    "approach": "bi_encoder_plus_cross_encoder",
                    "index_build_ms": dense_output.corpus_embedding_ms,
                    "query_embedding_ms": dense_output.query_embedding_ms,
                    "search_ms": dense_output.search_ms,
                    "reranking_ms": reranked_output.reranking_ms,
                    "pair_count": reranked_output.pair_count,
                    "mean_query_ms": (
                        dense_output.mean_query_ms + reranked_output.mean_query_ms
                    ),
                    "device": reranked_output.device,
                }
            )

            before = dense_details.set_index("query_id")
            after = reranked_details.set_index("query_id")
            bootstrap_results[dataset.name] = {
                "mrr_at_10": paired_bootstrap_delta(
                    before["mrr_at_10"].to_numpy(),
                    after["mrr_at_10"].to_numpy(),
                    samples=config.bootstrap_samples,
                    seed=config.random_seed,
                ),
                "ndcg_at_10": paired_bootstrap_delta(
                    before["ndcg_at_10"].to_numpy(),
                    after["ndcg_at_10"].to_numpy(),
                    samples=config.bootstrap_samples,
                    seed=config.random_seed,
                ),
                "map_at_100": paired_bootstrap_delta(
                    before["map_at_100"].to_numpy(),
                    after["map_at_100"].to_numpy(),
                    samples=config.bootstrap_samples,
                    seed=config.random_seed,
                ),
            }
            examples = _ranking_examples(
                dataset,
                dense_output.rankings,
                reranked_output.rankings,
                dense_details,
                reranked_details,
            )
            all_examples.append(examples)

        del dense
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = pd.DataFrame(summary_rows)
    latency = pd.DataFrame(latency_rows)
    per_query = pd.concat(all_per_query, ignore_index=True)
    examples = (
        pd.concat(all_examples, ignore_index=True)
        if all_examples
        else pd.DataFrame()
    )

    # Add paired delta rows for visualization.
    rerank_per_query: list[pd.DataFrame] = []
    for dataset_name in config.datasets:
        before = per_query[
            (per_query["dataset"] == dataset_name)
            & (per_query["approach"] == "bi_encoder")
        ].set_index("query_id")
        after = per_query[
            (per_query["dataset"] == dataset_name)
            & (per_query["approach"] == "bi_encoder_plus_cross_encoder")
        ].set_index("query_id")
        if before.empty or after.empty:
            continue
        common = before.index.intersection(after.index)
        frame = pd.DataFrame(
            {
                "dataset": dataset_name,
                "query_id": common,
                "mrr_delta": after.loc[common, "mrr_at_10"].to_numpy()
                - before.loc[common, "mrr_at_10"].to_numpy(),
                "ndcg_delta": after.loc[common, "ndcg_at_10"].to_numpy()
                - before.loc[common, "ndcg_at_10"].to_numpy(),
                "map_delta": after.loc[common, "map_at_100"].to_numpy()
                - before.loc[common, "map_at_100"].to_numpy(),
            }
        )
        rerank_per_query.append(frame)
    delta_frame = (
        pd.concat(rerank_per_query, ignore_index=True)
        if rerank_per_query
        else pd.DataFrame()
    )

    summary.to_csv(run_dir / "benchmark_summary.csv", index=False)
    latency.to_csv(run_dir / "latency_breakdown.csv", index=False)
    per_query.to_csv(run_dir / "per_query_metrics.csv", index=False)
    examples.to_csv(run_dir / "ranking_examples.csv", index=False)
    delta_frame.to_csv(run_dir / "reranking_deltas.csv", index=False)
    write_json(run_dir / "benchmark_summary.json", summary.to_dict(orient="records"))
    write_json(run_dir / "latency_breakdown.json", latency.to_dict(orient="records"))
    write_json(run_dir / "bootstrap_significance.json", bootstrap_results)
    write_json(run_dir / "dataset_metadata.json", metadata)

    plot_metric_comparison(summary, run_dir / "metric_comparison.png")
    plot_recall_curves(summary, run_dir / "recall_at_k_curves.png")
    plot_latency(latency, run_dir / "latency_comparison.png")
    if not delta_frame.empty:
        plot_reranking_delta(delta_frame, run_dir / "reranking_delta_distribution.png")

    portfolio_markdown = make_portfolio_markdown(summary, bootstrap_results, metadata)
    (run_dir / "PORTFOLIO_RESULTS.md").write_text(
        portfolio_markdown,
        encoding="utf-8",
    )
    latest_dir = update_latest_alias(run_dir, config.benchmark_output_root)

    return BenchmarkRunResult(
        run_directory=run_dir,
        latest_directory=latest_dir,
        summary=summary,
        latency=latency,
        per_query=per_query,
        bootstrap=bootstrap_results,
        metadata=metadata,
    )
