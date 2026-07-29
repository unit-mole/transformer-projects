from __future__ import annotations

import csv
import hashlib
import json
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BEIR_DOWNLOAD_BASE = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"
)

# MD5 values published by the official BEIR repository.
KNOWN_MD5 = {
    "scifact": "5f7d1de60b170fc8027bb7898e2efca1",
    "nfcorpus": "a89dba18a62ef92f7d323ec890a0d38d",
}


@dataclass(frozen=True)
class BEIRDataset:
    name: str
    split: str
    corpus: dict[str, dict[str, str]]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]
    source_directory: Path

    @property
    def corpus_ids(self) -> list[str]:
        return list(self.corpus)

    @property
    def corpus_texts(self) -> list[str]:
        return [
            combine_title_and_text(
                self.corpus[document_id].get("title", ""),
                self.corpus[document_id].get("text", ""),
            )
            for document_id in self.corpus_ids
        ]

    @property
    def query_ids(self) -> list[str]:
        return list(self.queries)

    @property
    def query_texts(self) -> list[str]:
        return [self.queries[query_id] for query_id in self.query_ids]

    def metadata(self) -> dict[str, Any]:
        relevance_count = sum(len(documents) for documents in self.qrels.values())
        relevant_per_query = [len(documents) for documents in self.qrels.values()]
        return {
            "dataset": self.name,
            "split": self.split,
            "corpus_documents": len(self.corpus),
            "queries": len(self.queries),
            "relevance_judgments": relevance_count,
            "average_relevant_documents_per_query": (
                sum(relevant_per_query) / len(relevant_per_query)
                if relevant_per_query
                else 0.0
            ),
            "source_directory": str(self.source_directory),
            "source": "BEIR benchmark",
        }


def combine_title_and_text(title: str, text: str) -> str:
    title = str(title or "").strip()
    text = str(text or "").strip()
    if title and text:
        return f"{title}. {text}"
    return title or text


def _md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()  # noqa: S324 - used only for published file verification.
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download_beir_dataset(
    dataset_name: str,
    cache_dir: str | Path,
    *,
    verify_md5: bool = True,
) -> Path:
    """Download and extract one official BEIR dataset into a local cache."""
    dataset_name = dataset_name.strip().lower()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    extracted_dir = cache_dir / dataset_name
    required = [
        extracted_dir / "corpus.jsonl",
        extracted_dir / "queries.jsonl",
        extracted_dir / "qrels",
    ]
    if all(path.exists() for path in required):
        return extracted_dir

    archive_path = cache_dir / f"{dataset_name}.zip"
    url = f"{BEIR_DOWNLOAD_BASE}/{dataset_name}.zip"

    if not archive_path.exists():
        print(f"Downloading BEIR {dataset_name} from {url}")
        temporary = archive_path.with_suffix(".zip.part")
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "DocRank360-BEIR-Benchmark/1.0"},
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310 - fixed official URL.
                with temporary.open("wb") as output:
                    shutil.copyfileobj(response, output)
            temporary.replace(archive_path)
        except Exception as exc:
            raise RuntimeError(
                f"Could not download {dataset_name} from the official BEIR URL. "
                "Check internet access, proxy settings and firewall rules."
            ) from exc
        finally:
            temporary.unlink(missing_ok=True)

    expected_md5 = KNOWN_MD5.get(dataset_name)
    if verify_md5 and expected_md5:
        actual_md5 = _md5(archive_path)
        if actual_md5 != expected_md5:
            archive_path.unlink(missing_ok=True)
            raise ValueError(
                f"MD5 mismatch for {archive_path.name}: "
                f"expected {expected_md5}, received {actual_md5}."
            )

    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(cache_dir)

    if not all(path.exists() for path in required):
        # Some archives can contain an unexpected wrapper directory. Search once
        # and normalize it to cache_dir/dataset_name.
        candidates = list(cache_dir.glob(f"**/{dataset_name}/corpus.jsonl"))
        if candidates:
            discovered = candidates[0].parent
            if discovered != extracted_dir:
                if extracted_dir.exists():
                    shutil.rmtree(extracted_dir)
                shutil.move(str(discovered), str(extracted_dir))

    if not all(path.exists() for path in required):
        raise FileNotFoundError(
            f"The extracted BEIR dataset is incomplete: {extracted_dir}"
        )
    return extracted_dir


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return rows


def _load_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            query_id = str(
                row.get("query-id")
                or row.get("query_id")
                or row.get("query")
                or ""
            ).strip()
            corpus_id = str(
                row.get("corpus-id")
                or row.get("corpus_id")
                or row.get("doc_id")
                or ""
            ).strip()
            score_text = row.get("score") or row.get("relevance") or "0"
            if not query_id or not corpus_id:
                continue
            score = int(float(score_text))
            if score > 0:
                qrels.setdefault(query_id, {})[corpus_id] = score
    return qrels


def load_beir_dataset(
    dataset_name: str,
    cache_dir: str | Path,
    *,
    split: str = "test",
    max_queries: int | None = None,
    verify_md5: bool = True,
) -> BEIRDataset:
    """Load a BEIR corpus, query set and qrels in deterministic query order."""
    data_dir = download_beir_dataset(
        dataset_name,
        cache_dir,
        verify_md5=verify_md5,
    )

    corpus_rows = _read_jsonl(data_dir / "corpus.jsonl")
    query_rows = _read_jsonl(data_dir / "queries.jsonl")
    qrels_path = data_dir / "qrels" / f"{split}.tsv"
    if not qrels_path.exists():
        available = sorted(path.stem for path in (data_dir / "qrels").glob("*.tsv"))
        raise FileNotFoundError(
            f"Split '{split}' is unavailable for {dataset_name}. Available: {available}"
        )

    corpus = {
        str(row["_id"]): {
            "title": str(row.get("title", "") or ""),
            "text": str(row.get("text", "") or ""),
        }
        for row in corpus_rows
    }
    all_queries = {
        str(row["_id"]): str(row.get("text", "") or "")
        for row in query_rows
    }
    all_qrels = _load_qrels(qrels_path)

    # Evaluate only queries that have at least one relevance judgment.
    query_ids = sorted(query_id for query_id in all_qrels if query_id in all_queries)
    if max_queries is not None:
        query_ids = query_ids[: max(1, int(max_queries))]

    queries = {query_id: all_queries[query_id] for query_id in query_ids}
    qrels = {query_id: all_qrels[query_id] for query_id in query_ids}

    return BEIRDataset(
        name=dataset_name,
        split=split,
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        source_directory=data_dir,
    )
