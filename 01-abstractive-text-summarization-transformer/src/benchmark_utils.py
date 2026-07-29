from __future__ import annotations

import gc
import json
import math
import platform
import re
import shutil
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .baselines import lead3_summary, textrank_summary
from .model_evaluation import compute_rouge
from .text_preprocessing import word_count


@dataclass(frozen=True)
class GenerationConfig:
    max_source_tokens: int = 768
    min_new_tokens: int = 30
    max_new_tokens: int = 120
    num_beams: int = 4
    length_penalty: float = 2.0
    no_repeat_ngram_size: int = 3
    batch_size: int = 2


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def save_json(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def hardware_report() -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cuda_available": False,
    }
    try:
        import torch
        result.update({
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_runtime": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "gpu_count": torch.cuda.device_count(),
        })
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            result.update({
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_gb": round(props.total_memory / 1024**3, 2),
                "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            })
    except Exception as exc:
        result["torch_error"] = str(exc)
    return result


def extract_numbers(text: str) -> set[str]:
    return set(re.findall(r"(?<!\w)[+-]?\d+(?:[.,]\d+)*(?:%|st|nd|rd|th)?", str(text)))


def number_recall(prediction: str, reference: str) -> float:
    expected = extract_numbers(reference)
    if not expected:
        return 1.0
    return len(expected & extract_numbers(prediction)) / len(expected)


def hallucinated_numbers(prediction: str, article: str) -> int:
    return len(extract_numbers(prediction) - extract_numbers(article))


def repeated_trigram_ratio(text: str) -> float:
    tokens = re.findall(r"\b[\w'-]+\b", str(text).lower())
    if len(tokens) < 3:
        return 0.0
    grams = [tuple(tokens[i:i+3]) for i in range(len(tokens)-2)]
    return 1.0 - len(set(grams)) / len(grams)


def percentile(values: Sequence[float], q: float) -> float:
    vals = sorted(float(v) for v in values)
    if not vals:
        return 0.0
    pos = (len(vals)-1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return vals[lo]
    return vals[lo] + (vals[hi]-vals[lo]) * (pos-lo)


def run_baseline(articles: Sequence[str], method: str) -> tuple[list[str], list[float]]:
    fn = lead3_summary if method == "lead3" else textrank_summary if method == "textrank" else None
    if fn is None:
        raise ValueError("method must be lead3 or textrank")
    predictions, latencies = [], []
    for article in articles:
        start = time.perf_counter()
        predictions.append(fn(str(article), max_sentences=3))
        latencies.append(time.perf_counter() - start)
    return predictions, latencies


def generate_batched(model: Any, tokenizer: Any, articles: Sequence[str], config: GenerationConfig) -> dict[str, Any]:
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    summaries, latencies, batch_sizes = [], [], []
    peak_mb = 0.0
    offset = 0
    current_bs = max(1, int(config.batch_size))
    while offset < len(articles):
        batch = [str(x) for x in articles[offset:offset+current_bs]]
        try:
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=config.max_source_tokens, return_tensors="pt")
            encoded = {k: v.to(device) for k, v in encoded.items()}
            if use_amp:
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                generated = model.generate(
                    **encoded,
                    min_new_tokens=config.min_new_tokens,
                    max_new_tokens=config.max_new_tokens,
                    num_beams=config.num_beams,
                    length_penalty=config.length_penalty,
                    no_repeat_ngram_size=config.no_repeat_ngram_size,
                    early_stopping=True,
                    do_sample=False,
                )
            if use_amp:
                torch.cuda.synchronize()
                peak_mb = max(peak_mb, torch.cuda.max_memory_allocated()/1024**2)
            elapsed = time.perf_counter() - start
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            summaries.extend(x.strip() for x in decoded)
            latencies.extend([elapsed/len(batch)]*len(batch))
            batch_sizes.extend([len(batch)]*len(batch))
            offset += len(batch)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and current_bs > 1:
                current_bs = max(1, current_bs//2)
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                continue
            raise
    return {"summaries": summaries, "latencies": latencies, "batch_sizes": batch_sizes, "peak_gpu_memory_mb": peak_mb, "device": str(device)}


def per_sample_rouge(predictions: Sequence[str], references: Sequence[str]) -> pd.DataFrame:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rows = []
    for pred, ref in zip(predictions, references):
        score = scorer.score(str(ref), str(pred))
        rows.append({m: score[m].fmeasure for m in ("rouge1", "rouge2", "rougeL")})
    return pd.DataFrame(rows)


def evaluate_method(frame: pd.DataFrame, name: str, prediction_col: str, latency_col: str | None, bert_prefix: str | None) -> tuple[dict[str, Any], pd.DataFrame]:
    predictions = frame[prediction_col].fillna("").astype(str).tolist()
    references = frame["reference_summary"].fillna("").astype(str).tolist()
    details = pd.concat([
        frame[["id", "article", "reference_summary", prediction_col]].reset_index(drop=True),
        per_sample_rouge(predictions, references)
    ], axis=1)
    details["generated_words"] = [word_count(x) for x in predictions]
    details["reference_words"] = [word_count(x) for x in references]
    details["article_words"] = [word_count(x) for x in frame["article"].astype(str)]
    details["compression_ratio"] = details["generated_words"] / details["article_words"].clip(lower=1)
    details["reference_number_recall"] = [number_recall(p, r) for p, r in zip(predictions, references)]
    details["hallucinated_number_count"] = [hallucinated_numbers(p, a) for p, a in zip(predictions, frame["article"].astype(str))]
    details["repeated_trigram_ratio"] = [repeated_trigram_ratio(x) for x in predictions]
    metrics: dict[str, Any] = {"status": "completed", "model": name, "samples": len(frame), **compute_rouge(predictions, references)}
    metrics.update({
        "average_generated_words": float(details["generated_words"].mean()),
        "average_compression_ratio": float(details["compression_ratio"].mean()),
        "average_reference_number_recall": float(details["reference_number_recall"].mean()),
        "average_hallucinated_number_count": float(details["hallucinated_number_count"].mean()),
        "average_repeated_trigram_ratio": float(details["repeated_trigram_ratio"].mean()),
    })
    if latency_col and latency_col in frame.columns:
        vals = frame[latency_col].dropna().astype(float).tolist()
        details["inference_seconds"] = frame[latency_col].values
        metrics.update({
            "average_inference_seconds": statistics.fmean(vals) if vals else None,
            "minimum_inference_seconds": min(vals) if vals else None,
            "maximum_inference_seconds": max(vals) if vals else None,
            "p50_inference_seconds": percentile(vals, .5) if vals else None,
            "p95_inference_seconds": percentile(vals, .95) if vals else None,
        })
    else:
        metrics.update({k: None for k in ("average_inference_seconds","minimum_inference_seconds","maximum_inference_seconds","p50_inference_seconds","p95_inference_seconds")})
    if bert_prefix:
        for metric in ("precision", "recall", "f1"):
            col = f"{bert_prefix}_bertscore_{metric}"
            details[f"bertscore_{metric}"] = frame[col].values
            metrics[f"bertscore_{metric}"] = float(frame[col].mean())
    return metrics, details


def build_error_analysis(details: pd.DataFrame) -> pd.DataFrame:
    result = details.copy()
    low = float(result["rougeL"].quantile(.25)); high = float(result["rougeL"].quantile(.75))
    def classify(row: pd.Series) -> tuple[str, str]:
        tags = []
        if row.rougeL <= low: tags.append("low_rouge_l")
        if row.reference_number_recall < 1: tags.append("missing_reference_numbers")
        if row.hallucinated_number_count > 0: tags.append("possible_hallucinated_numbers")
        if row.repeated_trigram_ratio > .08: tags.append("repetition")
        if row.generated_words < max(12, row.reference_words*.45): tags.append("over_compression")
        if row.generated_words > row.reference_words*1.8: tags.append("overly_long")
        band = "strong" if row.rougeL >= high and not tags else "weak" if row.rougeL <= low or len(tags)>=2 else "mixed"
        return band, ";".join(tags) if tags else "none"
    labels = result.apply(classify, axis=1, result_type="expand")
    result["quality_band"] = labels[0]; result["error_tags"] = labels[1]
    return result


def comparison_frame(metrics_by_method: dict[str, dict[str, Any]]) -> pd.DataFrame:
    keys = ["samples","rouge1","rouge2","rougeL","bertscore_precision","bertscore_recall","bertscore_f1","average_inference_seconds","p95_inference_seconds","average_compression_ratio","average_reference_number_recall","average_hallucinated_number_count"]
    return pd.DataFrame([{"model": name, **{k: metrics.get(k) for k in keys}} for name, metrics in metrics_by_method.items()])


def write_error_report(path: str | Path, analysis: pd.DataFrame, prediction_col: str) -> Path:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Project 01 Error Analysis", "", "Generated from actual benchmark predictions. Human review is still required.", ""]
    counts = analysis["quality_band"].value_counts().rename_axis("quality_band").reset_index(name="count")
    lines += ["## Quality-band counts", "", counts.to_markdown(index=False), ""]
    for band in ("strong", "mixed", "weak"):
        subset = analysis[analysis.quality_band == band].sort_values("rougeL", ascending=(band=="weak")).head(5)
        lines += [f"## {band.title()} examples", ""]
        for _, row in subset.iterrows():
            trim = lambda x: re.sub(r"\s+", " ", str(x)).strip()[:420]
            lines += [
                f"### `{row.get('id','unknown')}`", "",
                f"- ROUGE-L: **{row.rougeL:.4f}**",
                f"- BERTScore F1: **{row.get('bertscore_f1', float('nan')):.4f}**" if pd.notna(row.get('bertscore_f1')) else "- BERTScore F1: not computed",
                f"- Error tags: `{row.error_tags}`", "",
                f"**Article:** {trim(row.article)}", "",
                f"**Reference:** {trim(row.reference_summary)}", "",
                f"**Generated:** {trim(row[prediction_col])}", "",
            ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def publish_outputs(project_root: str | Path, run_dir: str | Path, manifest: dict[str, Any], dataset_stats: dict[str, Any], training_summary: dict[str, Any], metrics_by_method: dict[str, dict[str, Any]], comparison: pd.DataFrame, predictions: pd.DataFrame, error_analysis: pd.DataFrame) -> None:
    root = Path(project_root); run = Path(run_dir); run.mkdir(parents=True, exist_ok=True)
    save_json(run/"benchmark_manifest.json", manifest)
    save_json(run/"dataset_statistics.json", dataset_stats)
    save_json(run/"training_summary.json", training_summary)
    save_json(run/"model_metrics.json", {"status":"completed","run_id":manifest["run_id"],"methods":metrics_by_method})
    save_json(run/"model_comparison.json", comparison.to_dict(orient="records"))
    predictions.to_csv(run/"all_predictions.csv", index=False)
    comparison.to_csv(run/"model_comparison.csv", index=False)
    error_analysis.to_csv(run/"error_analysis.csv", index=False)
    write_error_report(run/"error_analysis_examples.md", error_analysis, "fine_tuned_summary")
    display_cols = [c for c in ["model","samples","rouge1","rouge2","rougeL","bertscore_f1","average_inference_seconds","p95_inference_seconds"] if c in comparison]
    md = "# Project 01 Portfolio Benchmark Results\n\n" + comparison[display_cols].round(4).to_markdown(index=False) + "\n"
    (run/"PORTFOLIO_RESULTS.md").write_text(md, encoding="utf-8")

    latest = root/"outputs"/"benchmark"/"latest"
    if latest.exists(): shutil.rmtree(latest)
    shutil.copytree(run, latest)

    shutil.copy2(run/"model_metrics.json", root/"outputs"/"model_metrics.json")
    predictions.to_csv(root/"outputs"/"generated_summary_examples.csv", index=False)
    comparison.to_csv(root/"outputs"/"transformer_vs_lstm_comparison.csv", index=False)
    shutil.copy2(run/"error_analysis_examples.md", root/"outputs"/"error_analysis_examples.md")
    save_json(root/"outputs"/"rouge_scores.json", {m:{k:v.get(k) for k in ("rouge1","rouge2","rougeL")} for m,v in metrics_by_method.items()})
    save_json(root/"outputs"/"bertscore_results.json", {m:{k:v.get(k) for k in ("bertscore_precision","bertscore_recall","bertscore_f1")} for m,v in metrics_by_method.items()})
    save_json(root/"outputs"/"inference_time_results.json", {m:{k:v.get(k) for k in ("average_inference_seconds","minimum_inference_seconds","maximum_inference_seconds","p50_inference_seconds","p95_inference_seconds")} for m,v in metrics_by_method.items()})

    pretrained = metrics_by_method.get("Pretrained DistilBART", {})
    save_json(root/"web"/"public"/"evaluation-results.json", {
        "status":"completed", "evaluated_at":manifest["completed_at"], "dataset":manifest["dataset"]["id"],
        "sample_count":pretrained.get("samples"), "python_model":manifest["models"]["pretrained"],
        "browser_model":"Xenova/distilbart-cnn-12-6",
        "metrics":{"rouge_1":pretrained.get("rouge1"),"rouge_2":pretrained.get("rouge2"),"rouge_l":pretrained.get("rougeL"),"bertscore_f1":pretrained.get("bertscore_f1"),"average_inference_seconds":pretrained.get("average_inference_seconds")},
        "note":"Metrics match the base checkpoint used by the browser Static Space. Fine-tuned Python metrics are documented separately."
    })


def validate_outputs(project_root: str | Path, minimum_samples: int = 200) -> dict[str, Any]:
    root = Path(project_root); latest = root/"outputs"/"benchmark"/"latest"
    required = [latest/"benchmark_manifest.json",latest/"model_metrics.json",latest/"model_comparison.csv",latest/"all_predictions.csv",latest/"error_analysis.csv",latest/"PORTFOLIO_RESULTS.md",root/"web"/"public"/"evaluation-results.json"]
    missing = [str(p) for p in required if not p.is_file()]
    if missing: raise FileNotFoundError("Missing outputs:\n" + "\n".join(missing))
    manifest = json.loads((latest/"benchmark_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed": raise ValueError("Benchmark is not completed")
    samples = int(manifest.get("dataset",{}).get("test_samples",0))
    if samples < minimum_samples: raise ValueError(f"Only {samples} test samples")
    payload = json.loads((latest/"model_metrics.json").read_text(encoding="utf-8"))
    required_methods = {"Lead-3","TextRank","Pretrained DistilBART","Fine-tuned DistilBART"}
    methods = payload.get("methods",{})
    if required_methods - set(methods): raise ValueError("Required methods are missing")
    for method in required_methods:
        for metric in ("rouge1","rouge2","rougeL","bertscore_f1"):
            if methods[method].get(metric) is None: raise ValueError(f"{method} missing {metric}")
    return {"status":"valid","test_samples":samples,"methods":sorted(methods),"files_validated":len(required)}


def release_memory() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    except Exception:
        pass
