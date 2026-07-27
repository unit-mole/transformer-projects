import json
import numpy as np
from src.embedding_generator import save_browser_embeddings


def test_save_browser_embeddings(tmp_path):
    path = save_browser_embeddings(["a", "b"], np.array([[3.0, 4.0], [0.0, 2.0]]), tmp_path / "emb.json", model_id="test")
    payload = json.loads(path.read_text())
    assert payload["generated"] is True
    assert payload["dimension"] == 2
    assert len(payload["vectors"]) == 2
