from src.citation_evaluation import cited_ids

def test_extracts_unique_citation_ids():
    assert cited_ids("Claim [S1]. Another [S2] and [S1].") == ["S1", "S2"]
