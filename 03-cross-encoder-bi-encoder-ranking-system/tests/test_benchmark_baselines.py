from src.benchmarking.baselines import BM25Retriever, TfidfRetriever


def test_lexical_baselines_rank_matching_document_first():
    documents = [
        "cross encoder reranking improves semantic search",
        "satellite image segmentation with a vision transformer",
        "quality complaint retrieval and corrective action search",
    ]
    document_ids = ["A", "B", "C"]
    query_ids = ["Q1"]
    queries = ["quality complaint corrective action"]

    tfidf = TfidfRetriever()
    tfidf.fit(documents)
    tfidf_output = tfidf.search(query_ids, queries, document_ids, top_k=3)

    bm25 = BM25Retriever()
    bm25.fit(documents)
    bm25_output = bm25.search(query_ids, queries, document_ids, top_k=3)

    assert tfidf_output.rankings["Q1"][0] == "C"
    assert bm25_output.rankings["Q1"][0] == "C"
