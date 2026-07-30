from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.document_loader import load_documents
from src.text_preprocessing import clean_markdown
from src.document_chunking import chunk_markdown

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9+#.\-]{2,}")
REQUIRED_PORTFOLIO_CATEGORIES = {"ANN", "Simple RNN", "LSTM", "BiLSTM", "CNN", "Transformer"}

STOP_WORDS = {
    "and", "the", "for", "with", "from", "this", "that", "are", "was", "were", "into",
    "using", "used", "project", "projects", "portfolio", "readme", "model", "models", "data",
}


def infer_category(path: str, hint: str = "") -> str:
    lowered = f"{hint}/{path}".lower()
    if "bilstm" in lowered or "bi-directional" in lowered or "bidirectional" in lowered:
        return "BiLSTM"
    if "simple-rnn" in lowered or "/rnn" in lowered:
        return "Simple RNN"
    if "lstm" in lowered:
        return "LSTM"
    if "cnn" in lowered:
        return "CNN"
    if "ann" in lowered or "deep-learning" in lowered:
        return "ANN"
    if "transformer" in lowered:
        return "Transformer"
    return "Portfolio"


def infer_deployment(text: str, source_file: str) -> str:
    lowered = f"{source_file}\n{text}".lower()
    if "vercel" in lowered:
        return "Vercel"
    if "github pages" in lowered or "github-pages" in lowered:
        return "GitHub Pages"
    if "hugging face" in lowered or "huggingface" in lowered:
        return "Hugging Face"
    if "streamlit" in lowered:
        return "Streamlit"
    if "gradio" in lowered:
        return "Gradio"
    if "tensorflow.js" in lowered or "tensorflowjs" in lowered:
        return "TensorFlow.js"
    return "Not specified"


def title_from_project_id(project_id: str) -> str:
    title = re.sub(r"^\d+[-_]", "", project_id)
    replacements = {"rag": "RAG", "rnn": "RNN", "lstm": "LSTM", "cnn": "CNN", "ann": "ANN", "llm": "LLM", "qa": "QA", "vqa": "VQA", "clip": "CLIP"}
    words = []
    for word in title.replace("_", "-").split("-"):
        words.append(replacements.get(word.lower(), word.capitalize()))
    return " ".join(words)


def keywords(text: str, project_id: str, limit: int = 30) -> list[str]:
    tokens = [token.lower() for token in TOKEN_PATTERN.findall(text)]
    counts = Counter(token for token in tokens if token not in STOP_WORDS)
    selected = [token for token, _ in counts.most_common(limit)]
    return [project_id, *selected]


def repository_url(repository: str, source_path: str) -> str:
    parts = PurePosixPath(source_path).parts
    try:
        repo_index = parts.index(repository)
        relative = "/".join(parts[repo_index + 1 :])
    except ValueError:
        # Legacy starter layout is category/project/file rather than category/repository/project/file.
        relative_parts = parts[1:] if parts and parts[0].lower() in {
            "ann", "simple-rnn", "lstm", "bilstm", "cnn", "transformer", "portfolio"
        } else parts
        relative = "/".join(relative_parts)
    return f"https://github.com/unit-mole/{repository}/blob/main/{relative}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a section-aware portfolio corpus.")
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data/raw_portfolio_docs")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/processed")
    parser.add_argument("--chunk-size", type=int, default=220)
    parser.add_argument("--overlap", type=int, default=50)
    args = parser.parse_args()

    documents = load_documents(args.input)
    records: list[dict] = []
    chunks: list[dict] = []
    category_counts: Counter[str] = Counter()
    deployment_counts: Counter[str] = Counter()

    for document in documents:
        cleaned = clean_markdown(document.text)
        category = infer_category(document.source_path, document.category_hint)
        deployment = infer_deployment(cleaned, document.source_file)
        category_counts[category] += 1
        deployment_counts[deployment] += 1
        records.append({**document.to_dict(), "text": cleaned, "category": category, "deployment": deployment})

        for chunk in chunk_markdown(document.document_id, cleaned, args.chunk_size, args.overlap):
            text = chunk.text.strip()
            chunks.append(
                {
                    "id": chunk.chunk_id,
                    "projectId": document.project_id,
                    "projectName": title_from_project_id(document.project_id),
                    "category": category,
                    "deployment": deployment,
                    "sourceFile": document.source_file,
                    "section": chunk.section,
                    "sourcePath": document.source_path,
                    "repository": document.repository,
                    "repositoryUrl": repository_url(document.repository, document.source_path),
                    "text": text,
                    "keywords": keywords(f"{chunk.section} {text}", document.project_id),
                    "startWord": chunk.start_word,
                    "endWord": chunk.end_word,
                    "documentId": document.document_id,
                    "checksumSha256": document.checksum_sha256,
                }
            )

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "portfolio_corpus.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (args.output / "document_chunks.json").write_text(json.dumps(chunks, indent=2), encoding="utf-8")

    existing_metadata_path = args.output / "metadata.json"
    existing = json.loads(existing_metadata_path.read_text(encoding="utf-8")) if existing_metadata_path.exists() else {}
    metadata = {
        "schemaVersion": "2.0",
        "corpusName": "Anmol Tripathi public AI portfolio corpus",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "documentCount": len(documents),
        "chunkCount": len(chunks),
        "coverageStatus": (
            "complete"
            if REQUIRED_PORTFOLIO_CATEGORIES.issubset(set(category_counts))
            else "partial"
        ),
        "categories": sorted(category_counts),
        "deployments": sorted(deployment_counts),
        "categoryDocumentCounts": dict(sorted(category_counts.items())),
        "deploymentDocumentCounts": dict(sorted(deployment_counts.items())),
        "sourceRepositories": sorted({document.repository for document in documents}),
        "chunking": {
            "strategy": "section-aware Markdown chunking",
            "sizeWords": args.chunk_size,
            "overlapWords": args.overlap,
        },
        "embedding": existing.get(
            "embedding",
            {
                "provider": "pending",
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "dimension": 384,
                "normalized": True,
            },
        ),
        "retrieval": {
            "similarity": "cosine",
            "defaultTopK": 5,
            "hybridSemanticWeight": 0.78,
            "hybridLexicalWeight": 0.22,
        },
        "dataSafety": {
            "publicDocumentsOnly": True,
            "excluded": ["private company files", "emails", "GCS data", "proprietary documents", "PII"],
        },
        "notes": [
            "Every evaluation result must be regenerated after corpus changes.",
            "Document embeddings and query embeddings must use the same model and normalization settings.",
        ],
    }
    existing_metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    stats = {
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "category_document_counts": dict(category_counts),
        "deployment_document_counts": dict(deployment_counts),
        "mean_chunk_words": round(sum(len(chunk["text"].split()) for chunk in chunks) / len(chunks), 2) if chunks else 0,
        "min_chunk_words": min((len(chunk["text"].split()) for chunk in chunks), default=0),
        "max_chunk_words": max((len(chunk["text"].split()) for chunk in chunks), default=0),
    }
    outputs = PROJECT_ROOT / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    (outputs / "corpus_statistics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Prepared {len(documents)} documents and {len(chunks)} chunks.")


if __name__ == "__main__":
    main()
