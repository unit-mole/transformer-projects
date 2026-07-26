from __future__ import annotations

import numpy as np

from src.answer_extraction import select_best_span_from_arrays


def test_select_best_valid_context_span() -> None:
    context = "Alpha beta gamma delta."
    # <s> question </s></s> Alpha beta gamma delta </s>
    offsets = np.asarray(
        [
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 5],
            [6, 10],
            [11, 16],
            [17, 23],
            [0, 0],
        ]
    )
    sequence_ids = [None, 0, None, 1, 1, 1, 1, None]
    start_logits = np.asarray([-2, -2, -2, 0.1, 8.0, 0.2, 0.1, -2])
    end_logits = np.asarray([-2, -2, -2, 0.1, 0.2, 8.5, 0.1, -2])

    candidate = select_best_span_from_arrays(
        context=context,
        start_logits=start_logits,
        end_logits=end_logits,
        offsets=offsets,
        sequence_ids=sequence_ids,
        max_answer_tokens=4,
    )

    assert candidate is not None
    assert candidate.answer == "beta gamma"
    assert candidate.start_char == 6
    assert candidate.end_char == 16
    assert 0.0 <= candidate.confidence_proxy <= 1.0
