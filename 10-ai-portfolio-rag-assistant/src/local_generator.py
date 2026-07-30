from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Sequence


@dataclass(frozen=True)
class GeneratedAnswer:
    """Container for a generated answer and its runtime metadata."""

    text: str
    latency_ms: float
    mode: str


def _clean_text(value: str) -> str:
    """Normalize whitespace without removing technical information."""

    return " ".join(str(value).split())


def build_prompt(
    question: str,
    chunks: Sequence[dict],
    max_evidence_chars_per_chunk: int = 350,
) -> str:
    """
    Build a compact, source-cited prompt for FLAN-T5.

    The question is placed before the retrieved evidence so it remains
    visible even when the tokenizer must truncate a long prompt.
    """

    context_blocks: list[str] = []

    for index, chunk in enumerate(chunks, start=1):
        citation_id = f"S{index}"

        evidence = _clean_text(chunk.get("text", ""))

        if len(evidence) > max_evidence_chars_per_chunk:
            evidence = evidence[:max_evidence_chars_per_chunk].rsplit(" ", 1)[0]
            evidence += "..."

        project_name = _clean_text(
            chunk.get("projectName", "Unknown project")
        )
        source_file = _clean_text(
            chunk.get("sourceFile", "Unknown source")
        )
        section = _clean_text(
            chunk.get("section", "Unknown section")
        )

        context_blocks.append(
            f"[{citation_id}]\n"
            f"Project: {project_name}\n"
            f"Source: {source_file}\n"
            f"Section: {section}\n"
            f"Evidence: {evidence}"
        )

    context = "\n\n".join(context_blocks)

    return f"""You are an AI Portfolio Assistant for Anmol Tripathi.

Question:
{question}

Instructions:
1. Answer using only the retrieved portfolio evidence.
2. Do not invent projects, models, datasets, metrics, deployments, employers, or links.
3. Add a source citation such as [S1] after every factual claim.
4. Use only citation identifiers that appear in the retrieved evidence.
5. Be concise, accurate, professional, and recruiter-friendly.
6. If the evidence is insufficient, reply exactly:
I could not find enough supporting information in the indexed portfolio documents to answer this confidently.

Retrieved portfolio evidence:
{context}

Source-cited answer:
"""


class LocalInstructionGenerator:
    """
    Local FLAN-T5 generator using the Transformers sequence-to-sequence API.

    This implementation intentionally avoids the removed
    'text2text-generation' pipeline and calls model.generate() directly.
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        device: str | int | None = None,
        max_new_tokens: int = 192,
        max_input_tokens: int = 512,
        num_beams: int = 2,
    ) -> None:
        try:
            import torch
            from transformers import (
                AutoModelForSeq2SeqLM,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Install torch, transformers, sentencepiece, and protobuf "
                "before using local FLAN-T5 generation."
            ) from exc

        self.torch = torch
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.max_input_tokens = max_input_tokens
        self.num_beams = num_beams

        self.device = self._resolve_device(device)

        print(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False,
        )

        print(f"Loading generator model: {model_name}")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        self.model.to(self.device)
        self.model.eval()

        print(f"Generator device: {self.device}")

        if self.device.type == "cuda":
            print(
                "Generator GPU:",
                torch.cuda.get_device_name(self.device.index or 0),
            )

    def _resolve_device(
        self,
        device: str | int | None,
    ) -> Any:
        """Convert a user device setting into a torch.device."""

        torch = self.torch

        if device is None:
            return torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )

        if isinstance(device, int):
            if device < 0:
                return torch.device("cpu")

            if not torch.cuda.is_available():
                raise RuntimeError(
                    "A CUDA device was requested, but CUDA is unavailable."
                )

            return torch.device(f"cuda:{device}")

        normalized = str(device).strip().lower()

        if normalized == "cpu":
            return torch.device("cpu")

        if normalized == "cuda":
            normalized = "cuda:0"

        if normalized.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"Device {normalized} was requested, "
                    "but CUDA is unavailable."
                )

            return torch.device(normalized)

        raise ValueError(
            f"Unsupported generation device: {device}"
        )

    def generate(
        self,
        question: str,
        chunks: Sequence[dict],
    ) -> GeneratedAnswer:
        """Generate one grounded portfolio answer on the selected device."""

        prompt = build_prompt(question, chunks)

        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )

        encoded = {
            name: tensor.to(self.device)
            for name, tensor in encoded.items()
        }

        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

        start = perf_counter()

        with self.torch.inference_mode():
            output_ids = self.model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=self.num_beams,
                early_stopping=True,
            )

        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

        latency_ms = (perf_counter() - start) * 1000

        text = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()

        if not text:
            text = (
                "I could not find enough supporting information in the "
                "indexed portfolio documents to answer this confidently."
            )

        return GeneratedAnswer(
            text=text,
            latency_ms=latency_ms,
            mode=self.model_name,
        )


def grounded_extractive_answer(
    question: str,
    chunks: Sequence[dict],
    min_retrieval_score: float = 0.20,
) -> GeneratedAnswer:
    """
    Deterministic grounded baseline used when Transformer generation
    is disabled.
    """

    del question

    start = perf_counter()

    best_score = max(
        (
            float(chunk.get("_retrievalScore", 1.0))
            for chunk in chunks
        ),
        default=0.0,
    )

    if not chunks or best_score < min_retrieval_score:
        text = (
            "I could not find enough supporting information in the "
            "indexed portfolio documents to answer this confidently."
        )
    else:
        lines: list[str] = []

        for index, chunk in enumerate(chunks[:4], start=1):
            evidence = _clean_text(chunk.get("text", ""))

            first_sentence = evidence.split(".")[0].strip()

            if first_sentence:
                first_sentence += "."

            project_name = _clean_text(
                chunk.get("projectName", "Unknown project")
            )

            lines.append(
                f"- {project_name}: "
                f"{first_sentence} [S{index}]"
            )

        text = "\n".join(lines)

    return GeneratedAnswer(
        text=text,
        latency_ms=(perf_counter() - start) * 1000,
        mode="grounded-extractive",
    )