from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

from .confidence_scoring import span_confidence_proxy
from .schemas import SpanCandidate


def _top_indices(values: np.ndarray, valid_positions: list[int], n_best: int) -> list[int]:
    if not valid_positions:
        return []
    ordered = sorted(valid_positions, key=lambda index: float(values[index]), reverse=True)
    return ordered[: min(n_best, len(ordered))]


def select_best_span_from_arrays(
    context: str,
    start_logits: np.ndarray,
    end_logits: np.ndarray,
    offsets: Sequence[Sequence[int]],
    sequence_ids: Sequence[Optional[int]],
    feature_index: int = 0,
    max_answer_tokens: int = 48,
    n_best: int = 20,
) -> Optional[SpanCandidate]:
    """Select the highest-scoring valid context span for one token window."""
    valid_positions = [
        index
        for index, sequence_id in enumerate(sequence_ids)
        if sequence_id == 1
        and index < len(offsets)
        and len(offsets[index]) >= 2
        and int(offsets[index][1]) > int(offsets[index][0])
    ]
    if not valid_positions:
        return None

    starts = _top_indices(start_logits, valid_positions, n_best)
    ends = _top_indices(end_logits, valid_positions, n_best)

    best: Optional[SpanCandidate] = None
    for start_index in starts:
        for end_index in ends:
            if end_index < start_index:
                continue
            if end_index - start_index + 1 > max_answer_tokens:
                continue
            start_char = int(offsets[start_index][0])
            end_char = int(offsets[end_index][1])
            if start_char < 0 or end_char <= start_char or end_char > len(context):
                continue
            answer = context[start_char:end_char].strip()
            if not answer:
                continue
            raw_score = float(start_logits[start_index] + end_logits[end_index])
            proxy = span_confidence_proxy(
                start_logits=start_logits,
                end_logits=end_logits,
                valid_positions=valid_positions,
                start_index=start_index,
                end_index=end_index,
            )
            candidate = SpanCandidate(
                answer=answer,
                start_char=start_char,
                end_char=end_char,
                raw_score=raw_score,
                confidence_proxy=proxy,
                feature_index=feature_index,
                start_token=start_index,
                end_token=end_index,
            )
            if best is None or candidate.raw_score > best.raw_score:
                best = candidate
    return best


def select_best_span_across_features(
    context: str,
    start_logits_batch: np.ndarray,
    end_logits_batch: np.ndarray,
    offsets_batch: np.ndarray,
    sequence_id_batch: Sequence[Sequence[Optional[int]]],
    max_answer_tokens: int = 48,
    n_best: int = 20,
) -> tuple[Optional[SpanCandidate], list[SpanCandidate]]:
    candidates: list[SpanCandidate] = []
    feature_count = int(start_logits_batch.shape[0])
    for feature_index in range(feature_count):
        candidate = select_best_span_from_arrays(
            context=context,
            start_logits=np.asarray(start_logits_batch[feature_index]),
            end_logits=np.asarray(end_logits_batch[feature_index]),
            offsets=np.asarray(offsets_batch[feature_index]),
            sequence_ids=sequence_id_batch[feature_index],
            feature_index=feature_index,
            max_answer_tokens=max_answer_tokens,
            n_best=n_best,
        )
        if candidate is not None:
            candidates.append(candidate)

    if not candidates:
        return None, []
    best = max(candidates, key=lambda candidate: candidate.raw_score)
    return best, candidates
