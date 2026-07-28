from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.portfolio_evaluation import (  # noqa: E402
    collect_environment,
    compare_systems,
    create_manual_review_candidates,
    create_plots,
    detect_hardware,
    evaluate_system,
    fine_tune_both_directions,
    fine_tuned_model_refs,
    load_and_prepare_dataset,
    load_config,
    load_prepared_dataset,
    save_comparison_artifacts,
    save_prepared_dataset,
    set_reproducibility,
    summarize_manual_review,
    sync_portfolio_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Portfolio-grade MarianMT fine-tuning and evaluation pipeline."
    )
    parser.add_argument(
        "--stage",
        choices=[
            "prepare",
            "pretrained",
            "finetune",
            "fine-tuned",
            "compare",
            "manual-template",
            "manual-summary",
            "sync",
            "all",
        ],
        default="all",
    )
    parser.add_argument("--profile", choices=["quick", "portfolio", "full"], default=None)
    return parser.parse_args()


def output_root(config: dict) -> Path:
    return PROJECT_ROOT / config["outputs"]["root"]


def main() -> None:
    args = parse_args()
    config = load_config(PROJECT_ROOT, profile=args.profile)
    set_reproducibility(int(config["seed"]))
    hardware = detect_hardware()
    environment = collect_environment(hardware)
    print(json.dumps({"profile": config["active_profile"], "hardware": hardware.__dict__, "environment": environment}, indent=2))

    stages = (
        ["prepare", "pretrained", "finetune", "fine-tuned", "compare", "manual-template"]
        if args.stage == "all"
        else [args.stage]
    )

    frames = None
    comparison = None
    comparison_summary = None

    for stage in stages:
        if stage == "prepare":
            frames = load_and_prepare_dataset(config)
            print(save_prepared_dataset(frames, config, environment))

        elif stage == "pretrained":
            frames = frames or load_prepared_dataset(config)
            print(
                json.dumps(
                    evaluate_system(
                        frames["test"],
                        system_name="pretrained",
                        model_refs=config["models"],
                        hardware=hardware,
                        config=config,
                    ),
                    indent=2,
                )
            )

        elif stage == "finetune":
            frames = frames or load_prepared_dataset(config)
            print(json.dumps(fine_tune_both_directions(frames, hardware, config), indent=2))

        elif stage == "fine-tuned":
            frames = frames or load_prepared_dataset(config)
            print(
                json.dumps(
                    evaluate_system(
                        frames["test"],
                        system_name="fine_tuned",
                        model_refs=fine_tuned_model_refs(config),
                        hardware=hardware,
                        config=config,
                    ),
                    indent=2,
                )
            )

        elif stage == "compare":
            comparison, comparison_summary = compare_systems(config)
            print(save_comparison_artifacts(comparison, comparison_summary, config))
            print(create_plots(comparison, config))

        elif stage == "manual-template":
            candidates = create_manual_review_candidates(config)
            path = output_root(config) / "manual_error_analysis_candidates.csv"
            candidates.to_csv(path, index=False, encoding="utf-8-sig")
            print(f"Created {path}")

        elif stage == "manual-summary":
            path = output_root(config) / "manual_error_analysis_candidates.csv"
            review = __import__("pandas").read_csv(path)
            summary = summarize_manual_review(review)
            target = output_root(config) / "manual_error_analysis_summary.json"
            target.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(summary, ensure_ascii=False, indent=2))

        elif stage == "sync":
            comparison, comparison_summary = compare_systems(config)
            manual_path = output_root(config) / "manual_error_analysis_candidates.csv"
            if manual_path.exists():
                review = __import__("pandas").read_csv(manual_path)
                manual_summary = summarize_manual_review(review)
            else:
                manual_summary = {"status": "not_created", "reviewed_examples": 0}
            print(sync_portfolio_outputs(comparison, comparison_summary, manual_summary, config))


if __name__ == "__main__":
    main()
