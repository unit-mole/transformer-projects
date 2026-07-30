from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = PROJECT_ROOT / "web"

REQUIRED_FILES = [
    WEB_ROOT / "index.html",
    WEB_ROOT / "package.json",
    WEB_ROOT / "vite.config.js",
    WEB_ROOT / "public" / "README.md",
    WEB_ROOT / "public" / "data" / "sample_documents.json",
    WEB_ROOT / "public" / "data" / "sample_queries.json",
    WEB_ROOT / "public" / "data" / "sample_qrels.json",
    WEB_ROOT / "public" / "data" / "benchmark_summary.json",
    WEB_ROOT / "src" / "constants.js",
    WEB_ROOT / "src" / "data-loader.js",
    WEB_ROOT / "src" / "metrics.js",
    WEB_ROOT / "src" / "export-results.js",
    WEB_ROOT / "src" / "ranking-engine.js",
    WEB_ROOT / "src" / "ui.js",
    WEB_ROOT / "src" / "main.js",
    WEB_ROOT / "src" / "styles.css",
]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    missing = [
        str(path.relative_to(PROJECT_ROOT))
        for path in REQUIRED_FILES
        if not path.exists()
    ]
    if missing:
        raise SystemExit(f"Missing browser files: {missing}")

    space_readme = (WEB_ROOT / "public" / "README.md").read_text(
        encoding="utf-8"
    )
    if not space_readme.startswith("---\n"):
        raise SystemExit("web/public/README.md must start with YAML metadata.")
    if "sdk: static" not in space_readme:
        raise SystemExit("Static Space metadata is missing: sdk: static")
    if "app_file: index.html" not in space_readme:
        raise SystemExit(
            "Static Space metadata is missing: app_file: index.html"
        )
    if "sdk: gradio" in space_readme:
        raise SystemExit("The Static Space card still declares Gradio.")

    documents = load_json(
        WEB_ROOT / "public" / "data" / "sample_documents.json"
    )
    queries = load_json(
        WEB_ROOT / "public" / "data" / "sample_queries.json"
    )
    qrels = load_json(
        WEB_ROOT / "public" / "data" / "sample_qrels.json"
    )
    benchmark = load_json(
        WEB_ROOT / "public" / "data" / "benchmark_summary.json"
    )

    if len(documents) != 24:
        raise SystemExit(f"Expected 24 documents, found {len(documents)}.")
    if len(queries) != 12:
        raise SystemExit(f"Expected 12 queries, found {len(queries)}.")
    if len(qrels) != 36:
        raise SystemExit(f"Expected 36 qrels, found {len(qrels)}.")

    if benchmark.get("status") != "completed":
        raise SystemExit("Benchmark summary must have status=completed.")
    if benchmark.get("totals", {}).get("queries") != 623:
        raise SystemExit("Expected 623 total benchmark queries.")
    if benchmark.get("totals", {}).get("documents") != 8816:
        raise SystemExit("Expected 8,816 total benchmark documents.")

    benchmark_datasets = {
        row.get("id"): row for row in benchmark.get("datasets", [])
    }
    expected_benchmark_datasets = {
        "scifact": 300,
        "nfcorpus": 323,
    }
    if set(benchmark_datasets) != set(expected_benchmark_datasets):
        raise SystemExit(
            "Benchmark summary must contain SciFact and NFCorpus."
        )

    for dataset_id, expected_queries in expected_benchmark_datasets.items():
        row = benchmark_datasets[dataset_id]
        if row.get("query_count") != expected_queries:
            raise SystemExit(
                f"Unexpected query count for {dataset_id}."
            )

        before = float(row["bi_encoder"]["ndcg_at_10"])
        after = float(row["reranked"]["ndcg_at_10"])
        delta = float(row["improvement"]["ndcg_at_10"]["absolute"])

        if abs((after - before) - delta) > 1e-9:
            raise SystemExit(
                f"nDCG improvement is inconsistent for {dataset_id}."
            )
        if delta <= 0:
            raise SystemExit(
                f"Expected positive nDCG improvement for {dataset_id}."
            )


    document_ids = {row["document_id"] for row in documents}
    query_ids = {row["query_id"] for row in queries}

    for row in qrels:
        if row["document_id"] not in document_ids:
            raise SystemExit(f"Unknown document ID: {row['document_id']}")
        if row["query_id"] not in query_ids:
            raise SystemExit(f"Unknown query ID: {row['query_id']}")
        if int(row["relevance"]) not in {1, 2, 3}:
            raise SystemExit(f"Invalid relevance: {row['relevance']}")

    package = load_json(WEB_ROOT / "package.json")
    dependency = package.get("dependencies", {}).get(
        "@huggingface/transformers"
    )
    if not dependency:
        raise SystemExit("@huggingface/transformers is missing.")

    ranking_engine = (
        WEB_ROOT / "src" / "ranking-engine.js"
    ).read_text(encoding="utf-8")
    for model_id in [
        "MODEL_IDS.biEncoder",
        "MODEL_IDS.crossEncoder",
        'dtype: "q8"',
    ]:
        if model_id not in ranking_engine:
            raise SystemExit(
                f"Expected browser inference configuration not found: {model_id}"
            )

    print("Project 03 web validation passed.")
    print("SDK: static")
    print("Build system: Vite")
    print("Documents: 24")
    print("Queries: 12")
    print("Qrels: 36")
    print("Benchmark queries: 623")
    print("Benchmark documents: 8,816")
    print("Browser runtime: Transformers.js + ONNX Runtime Web")


if __name__ == "__main__":
    main()
