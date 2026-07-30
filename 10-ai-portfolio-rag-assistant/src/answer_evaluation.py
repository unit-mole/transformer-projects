from __future__ import annotations

from dataclasses import dataclass, asdict
import re
from typing import Iterable, Mapping, Sequence

import numpy as np

CITATION_PATTERN = re.compile(r"\[(S\d+)\]")
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass(frozen=True)
class ClaimEvaluation:
    claim: str
    citation_ids: list[str]
    evidence_text: str
    entailment_score: float
    has_citation: bool
    supported: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AnswerEvaluation:
    question_id: str
    question: str
    answer: str
    claim_count: int
    grounded_claim_count: int
    cited_claim_count: int
    correct_citation_count: int
    groundedness_score: float
    citation_precision: float
    citation_completeness: float
    unsupported_claim_rate: float
    refusal_correct: bool | None
    claims: list[ClaimEvaluation]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["claims"] = [claim.to_dict() for claim in self.claims]
        return payload


def split_claims(answer: str) -> list[str]:
    claims: list[str] = []
    for part in SENTENCE_PATTERN.split(answer):
        text = part.strip(" -\t")
        if not text:
            continue
        if text.lower().startswith("based on the indexed portfolio evidence"):
            continue
        if text.lower().startswith("this answer is limited"):
            continue
        claims.append(text)
    return claims


def extract_citation_ids(text: str) -> list[str]:
    return sorted(set(CITATION_PATTERN.findall(text)))


class NliGroundednessEvaluator:
    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-small",
        device: str | None = None,
        entailment_threshold: float = 0.60,
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError("Install sentence-transformers before using NLI evaluation") from exc
        self.model_name = model_name
        self.entailment_threshold = entailment_threshold
        self.model = CrossEncoder(model_name, device=device)
        config = getattr(getattr(self.model, "model", None), "config", None)
        id2label = getattr(config, "id2label", {}) or {}
        self.labels = {int(index): str(label).lower() for index, label in id2label.items()}

    def _entailment_index(self, column_count: int) -> int:
        for index, label in self.labels.items():
            if "entail" in label:
                return index
        # The selected NLI model documents the order as contradiction, entailment, neutral.
        if column_count == 3:
            return 1
        raise RuntimeError(
            f"Could not determine entailment label for {self.model_name}; labels={self.labels}"
        )

    def _entailment_probabilities(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        if not pairs:
            return np.array([], dtype=np.float32)
        logits = np.asarray(self.model.predict(pairs, show_progress_bar=False), dtype=np.float32)
        if logits.ndim == 1:
            logits = logits.reshape(1, -1)
        logits = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs = probs / probs.sum(axis=1, keepdims=True)
        entailment_index = self._entailment_index(probs.shape[1])
        return probs[:, entailment_index]

    def evaluate(
        self,
        question_id: str,
        question: str,
        answer: str,
        citation_evidence: Mapping[str, str],
        all_retrieved_evidence: Iterable[str],
        answerable: bool | None = None,
    ) -> AnswerEvaluation:
        refusal_phrase = "could not find enough supporting information"
        refusal_detected = refusal_phrase in answer.lower()
        claims = [
            claim for claim in split_claims(answer)
            if refusal_phrase not in claim.lower()
        ]
        retrieved_text = "\n".join(text for text in all_retrieved_evidence if text)

        records: list[ClaimEvaluation] = []
        for claim in claims:
            citation_ids = extract_citation_ids(claim)
            cited_evidence = "\n".join(
                citation_evidence[cid] for cid in citation_ids if cid in citation_evidence
            ).strip()
            evidence = cited_evidence or retrieved_text
            score = 0.0
            if evidence:
                score = float(self._entailment_probabilities([(evidence, claim)])[0])
            records.append(
                ClaimEvaluation(
                    claim=claim,
                    citation_ids=citation_ids,
                    evidence_text=evidence,
                    entailment_score=round(score, 6),
                    has_citation=bool(citation_ids),
                    supported=score >= self.entailment_threshold,
                )
            )

        claim_count = len(records)
        grounded = sum(record.supported for record in records)
        cited = sum(record.has_citation for record in records)
        correct_citations = sum(record.has_citation and record.supported for record in records)

        groundedness = grounded / claim_count if claim_count else 1.0
        citation_precision = correct_citations / cited if cited else (1.0 if claim_count == 0 else 0.0)
        citation_completeness = cited / claim_count if claim_count else 1.0
        unsupported_rate = 1.0 - groundedness

        refusal_correct = None if answerable is None else (refusal_detected == (not answerable))

        return AnswerEvaluation(
            question_id=question_id,
            question=question,
            answer=answer,
            claim_count=claim_count,
            grounded_claim_count=grounded,
            cited_claim_count=cited,
            correct_citation_count=correct_citations,
            groundedness_score=round(groundedness, 6),
            citation_precision=round(citation_precision, 6),
            citation_completeness=round(citation_completeness, 6),
            unsupported_claim_rate=round(unsupported_rate, 6),
            refusal_correct=refusal_correct,
            claims=records,
        )


def summarize_answer_evaluations(rows: Sequence[AnswerEvaluation]) -> dict[str, float | int | None]:
    if not rows:
        return {
            "answer_count": 0,
            "mean_groundedness": 0.0,
            "mean_citation_precision": 0.0,
            "mean_citation_completeness": 0.0,
            "mean_unsupported_claim_rate": 0.0,
            "refusal_accuracy": None,
        }

    refusal_rows = [row for row in rows if row.refusal_correct is not None]
    return {
        "answer_count": len(rows),
        "mean_groundedness": round(sum(row.groundedness_score for row in rows) / len(rows), 6),
        "mean_citation_precision": round(sum(row.citation_precision for row in rows) / len(rows), 6),
        "mean_citation_completeness": round(sum(row.citation_completeness for row in rows) / len(rows), 6),
        "mean_unsupported_claim_rate": round(sum(row.unsupported_claim_rate for row in rows) / len(rows), 6),
        "refusal_accuracy": (
            round(sum(bool(row.refusal_correct) for row in refusal_rows) / len(refusal_rows), 6)
            if refusal_rows
            else None
        ),
    }
