from __future__ import annotations

import pytest

from src.release_utils import promote_experiment


def test_release_requires_human_review(tmp_path) -> None:
    with pytest.raises(ValueError):
        promote_experiment(project_root=tmp_path, experiment_dir=tmp_path / "run", human_review_completed=False)
