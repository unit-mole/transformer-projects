from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    print("\n$", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete Project 10 evaluation workflow.")
    parser.add_argument("--device", default=None, help="Examples: cuda, cuda:0, cpu")
    parser.add_argument("--generator", choices=["extractive", "flan-t5-base"], default="extractive")
    parser.add_argument("--include-e5", action="store_true")
    parser.add_argument("--include-reranker", action="store_true")
    args = parser.parse_args()

    python = sys.executable
    run([python, "scripts/prepare_corpus.py"])
    embedding_cmd = [python, "scripts/generate_embeddings.py", "--provider", "minilm"]
    if args.device:
        embedding_cmd += ["--device", args.device]
    run(embedding_cmd)
    run([python, "scripts/export_vector_store.py"])

    retrieval_cmd = [python, "scripts/run_retrieval_benchmark.py"]
    if args.device:
        retrieval_cmd += ["--device", args.device]
    if args.include_e5:
        retrieval_cmd.append("--include-e5")
    if args.include_reranker:
        retrieval_cmd.append("--include-reranker")
    run(retrieval_cmd)

    answer_cmd = [python, "scripts/evaluate_answers.py", "--generator", args.generator]
    if args.device:
        answer_cmd += ["--device", args.device]
    run(answer_cmd)

    run([python, "scripts/build_evaluation_summary.py"])
    run([python, "scripts/create_evaluation_charts.py"])
    print("\nComplete. Review outputs/ and public/data/evaluation_summary.json before committing.")


if __name__ == "__main__":
    main()
