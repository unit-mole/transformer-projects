from __future__ import annotations

import gc
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .answer_extraction import select_best_span_across_features
from .document_chunking import locate_supporting_paragraph


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    model_id: str
    strategy: str
    max_length: int
    stride: int = 0
    max_answer_tokens: int = 48
    inference_batch_size: int = 1
    description: str = ""

    def validate(self) -> "BenchmarkSpec":
        if self.strategy not in {"truncate", "sliding"}:
            raise ValueError("strategy must be 'truncate' or 'sliding'.")
        if self.max_length < 128:
            raise ValueError("max_length must be at least 128.")
        if self.strategy == "sliding" and not 0 <= self.stride < self.max_length:
            raise ValueError("stride must be smaller than max_length.")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TransformerQABenchmarkRunner:
    """Run extractive QA using either first-window truncation or sliding windows."""

    def __init__(
        self,
        spec: BenchmarkSpec,
        device: str = "auto",
        use_mixed_precision: bool = True,
    ) -> None:
        self.spec = spec.validate()
        self.requested_device = device
        self.use_mixed_precision = use_mixed_precision
        self._torch = None
        self.tokenizer = None
        self.model = None
        self.device = None
        self.model_type = ""

    def load(self) -> "TransformerQABenchmarkRunner":
        import torch
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.spec.model_id, use_fast=True)
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.spec.model_id)
        if self.requested_device == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device_name = self.requested_device
        self.device = torch.device(device_name)
        self.model.to(self.device)
        self.model.eval()
        self.model_type = str(getattr(self.model.config, "model_type", ""))
        return self

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
        gc.collect()
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    def _ensure_loaded(self) -> None:
        if self.model is None or self.tokenizer is None:
            self.load()

    def count_context_tokens(self, context: str) -> int:
        self._ensure_loaded()
        encoded = self.tokenizer(context, add_special_tokens=False)
        return int(len(encoded["input_ids"]))

    def answer_token_position(self, context: str, answer_start: int) -> int | None:
        self._ensure_loaded()
        encoded = self.tokenizer(
            context,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
        )
        for index, offset in enumerate(encoded["offset_mapping"]):
            if int(offset[0]) <= answer_start < int(offset[1]):
                return int(index)
        return None

    def _autocast_context(self):
        torch = self._torch
        if (
            self.use_mixed_precision
            and self.device.type == "cuda"
            and torch.cuda.is_available()
        ):
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            return torch.autocast(device_type="cuda", dtype=dtype)
        from contextlib import nullcontext

        return nullcontext()

    def predict(self, question: str, context: str) -> dict[str, Any]:
        self._ensure_loaded()
        torch = self._torch
        started = time.perf_counter()
        sliding = self.spec.strategy == "sliding"

        encoded = self.tokenizer(
            question.strip(),
            context,
            truncation="only_second",
            max_length=self.spec.max_length,
            stride=self.spec.stride if sliding else 0,
            return_overflowing_tokens=sliding,
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")
        encoded.pop("overflow_to_sample_mapping", None)
        sequence_ids = [encoded.sequence_ids(i) for i in range(encoded["input_ids"].shape[0])]

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        start_batches: list[np.ndarray] = []
        end_batches: list[np.ndarray] = []
        feature_count = int(encoded["input_ids"].shape[0])
        batch_size = max(1, self.spec.inference_batch_size)
        for start in range(0, feature_count, batch_size):
            end = min(start + batch_size, feature_count)
            model_inputs = {
                key: value[start:end].to(self.device)
                for key, value in encoded.items()
                if key in {"input_ids", "attention_mask", "token_type_ids"}
            }
            if self.model_type == "longformer":
                global_attention_mask = torch.zeros_like(model_inputs["attention_mask"])
                for local_index, feature_index in enumerate(range(start, end)):
                    ids = sequence_ids[feature_index]
                    for token_index, sequence_id in enumerate(ids):
                        if sequence_id == 0 and model_inputs["attention_mask"][local_index, token_index] == 1:
                            global_attention_mask[local_index, token_index] = 1
                model_inputs["global_attention_mask"] = global_attention_mask

            with torch.inference_mode(), self._autocast_context():
                outputs = self.model(**model_inputs)
            start_batches.append(outputs.start_logits.detach().float().cpu().numpy())
            end_batches.append(outputs.end_logits.detach().float().cpu().numpy())

        start_logits = np.concatenate(start_batches, axis=0)
        end_logits = np.concatenate(end_batches, axis=0)
        offsets_array = offsets.detach().cpu().numpy()
        candidate, candidates = select_best_span_across_features(
            context=context,
            start_logits_batch=start_logits,
            end_logits_batch=end_logits,
            offsets_batch=offsets_array,
            sequence_id_batch=sequence_ids,
            max_answer_tokens=self.spec.max_answer_tokens,
        )
        latency = time.perf_counter() - started
        peak_memory_mb = None
        if self.device.type == "cuda":
            peak_memory_mb = float(torch.cuda.max_memory_allocated(self.device) / 1024**2)

        if candidate is None:
            return {
                "predicted_answer": "",
                "predicted_evidence": "",
                "confidence_proxy": 0.0,
                "window_count": feature_count,
                "latency_seconds": latency,
                "peak_gpu_memory_mb": peak_memory_mb,
                "answer_start_char": None,
                "answer_end_char": None,
                "error": "no-valid-span",
            }

        paragraph, _ = locate_supporting_paragraph(
            context,
            candidate.start_char,
            candidate.end_char,
        )
        return {
            "predicted_answer": candidate.answer,
            "predicted_evidence": paragraph.text if paragraph else "",
            "confidence_proxy": float(candidate.confidence_proxy),
            "window_count": feature_count,
            "latency_seconds": latency,
            "peak_gpu_memory_mb": peak_memory_mb,
            "answer_start_char": int(candidate.start_char),
            "answer_end_char": int(candidate.end_char),
            "raw_span_score": float(candidate.raw_score),
            "candidate_count": len(candidates),
            "error": "",
        }


def evaluate_runner(
    runner: TransformerQABenchmarkRunner,
    examples: pd.DataFrame,
    progress_callback: Any | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    runner._ensure_loaded()
    for index, item in examples.reset_index(drop=True).iterrows():
        base = item.to_dict()
        try:
            token_count = runner.count_context_tokens(str(item["document"]))
            answer_position = runner.answer_token_position(
                str(item["document"]), int(item["answer_start"])
            )
            prediction = runner.predict(str(item["question"]), str(item["document"]))
        except Exception as exc:
            token_count = None
            answer_position = None
            prediction = {
                "predicted_answer": "",
                "predicted_evidence": "",
                "confidence_proxy": 0.0,
                "window_count": 0,
                "latency_seconds": 0.0,
                "peak_gpu_memory_mb": None,
                "answer_start_char": None,
                "answer_end_char": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
        base.update(prediction)
        base["model_name"] = runner.spec.name
        base["model_id"] = runner.spec.model_id
        base["strategy"] = runner.spec.strategy
        base["runtime_max_length"] = runner.spec.max_length
        base["runtime_stride"] = runner.spec.stride
        base["document_token_count"] = token_count
        base["answer_token_position"] = answer_position
        rows.append(base)
        if progress_callback:
            progress_callback(index + 1, len(examples), runner.spec.name)
    return pd.DataFrame(rows)
