from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any, Optional

import numpy as np

from .answer_extraction import select_best_span_across_features
from .config import InferenceConfig
from .schemas import SpanCandidate


class ModelLoadError(RuntimeError):
    """Raised when the Longformer model cannot be loaded."""


class LongformerQAModel:
    """Lazy-loading Longformer extractive QA wrapper.

    The wrapper performs overlapping token-window inference for documents that
    exceed the chosen runtime window and selects the best valid span across all
    windows. No training occurs during app startup.
    """

    def __init__(self, config: Optional[InferenceConfig] = None) -> None:
        self.config = (config or InferenceConfig()).validate()
        self.tokenizer: Any = None
        self.model: Any = None
        self.device: Any = None
        self.model_max_length: int = 4096

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def _resolve_device(self, torch_module: Any) -> Any:
        requested = self.config.device.lower()
        if requested == "auto":
            return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
        if requested == "cuda" and not torch_module.cuda.is_available():
            raise ModelLoadError("CUDA was requested, but no CUDA device is available.")
        return torch_module.device(requested)

    def load(self) -> None:
        if self.is_loaded:
            return
        if os.getenv("LONGDOCQA_SKIP_MODEL_LOAD", "0") == "1":
            raise ModelLoadError(
                "Model loading is disabled by LONGDOCQA_SKIP_MODEL_LOAD=1. "
                "Unset it for local or Hugging Face inference."
            )

        try:
            import torch
            from transformers import AutoModelForQuestionAnswering, AutoTokenizer
        except ImportError as exc:
            raise ModelLoadError(
                "PyTorch and Hugging Face Transformers are required for model inference. "
                "Install requirements.txt."
            ) from exc

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_id,
                use_fast=True,
            )
            model = AutoModelForQuestionAnswering.from_pretrained(
                self.config.model_id,
                low_cpu_mem_usage=True,
            )
        except Exception as exc:
            raise ModelLoadError(
                f"Unable to load {self.config.model_id!r} from Hugging Face Hub: {exc}"
            ) from exc

        self.device = self._resolve_device(torch)
        self.tokenizer = tokenizer
        self.model = model.to(self.device)
        self.model.eval()

        configured_max = getattr(model.config, "max_position_embeddings", 4098)
        # Longformer uses two reserved positions in this checkpoint.
        self.model_max_length = min(4096, max(512, int(configured_max) - 2))

    def _safe_runtime_window(self, requested: Optional[int]) -> int:
        value = int(requested or self.config.max_length)
        value = max(512, min(value, self.model_max_length, 4096))
        return value

    def count_context_tokens(self, context: str) -> int:
        self.load()
        encoded = self.tokenizer(
            context,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
        )
        return len(encoded["input_ids"])

    def predict(
        self,
        question: str,
        context: str,
        max_length: Optional[int] = None,
        stride: Optional[int] = None,
        max_answer_tokens: Optional[int] = None,
    ) -> dict[str, Any]:
        self.load()
        import torch

        runtime_max_length = self._safe_runtime_window(max_length)
        runtime_stride = int(stride if stride is not None else self.config.stride)
        runtime_stride = max(0, min(runtime_stride, runtime_max_length // 3))
        runtime_max_answer_tokens = int(
            max_answer_tokens or self.config.max_answer_tokens
        )

        encoded = self.tokenizer(
            question,
            context,
            truncation="only_second",
            max_length=runtime_max_length,
            stride=runtime_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_tensors="pt",
            padding=True,
        )
        sequence_id_batch = [
            encoded.sequence_ids(feature_index)
            for feature_index in range(len(encoded["input_ids"]))
        ]

        offsets = encoded.pop("offset_mapping")
        encoded.pop("overflow_to_sample_mapping", None)

        model_inputs = {
            key: value.to(self.device)
            for key, value in encoded.items()
            if hasattr(value, "to")
        }

        # Question tokens receive global attention. The first token also receives
        # global attention for stable Longformer QA behavior.
        global_attention_mask = torch.zeros_like(model_inputs["input_ids"])
        for feature_index, sequence_ids in enumerate(sequence_id_batch):
            global_attention_mask[feature_index, 0] = 1
            for token_index, sequence_id in enumerate(sequence_ids):
                if sequence_id == 0:
                    global_attention_mask[feature_index, token_index] = 1

        with torch.inference_mode():
            try:
                outputs = self.model(
                    **model_inputs,
                    global_attention_mask=global_attention_mask,
                )
            except TypeError:
                # Some compatible QA checkpoints set global attention internally.
                outputs = self.model(**model_inputs)

        start_logits = outputs.start_logits.detach().cpu().numpy()
        end_logits = outputs.end_logits.detach().cpu().numpy()
        offsets_array = offsets.detach().cpu().numpy()

        best, all_candidates = select_best_span_across_features(
            context=context,
            start_logits_batch=start_logits,
            end_logits_batch=end_logits,
            offsets_batch=offsets_array,
            sequence_id_batch=sequence_id_batch,
            max_answer_tokens=runtime_max_answer_tokens,
        )

        return {
            "candidate": best,
            "candidates": all_candidates,
            "window_count": int(start_logits.shape[0]),
            "runtime_max_length": runtime_max_length,
            "runtime_stride": runtime_stride,
            "model_max_length": self.model_max_length,
            "device": str(self.device),
        }
