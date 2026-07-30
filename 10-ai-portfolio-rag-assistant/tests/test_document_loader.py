from pathlib import Path
from src.document_loader import load_documents

def test_loads_non_empty_markdown(tmp_path: Path):
    project = tmp_path / "demo"
    project.mkdir()
    (project / "README.md").write_text("# Demo\n\nUseful content.", encoding="utf-8")
    docs = load_documents(tmp_path)
    assert len(docs) == 1
    assert docs[0].project_id == "demo"
