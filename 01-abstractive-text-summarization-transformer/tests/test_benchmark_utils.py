from __future__ import annotations
import pandas as pd
from src.benchmark_utils import build_error_analysis, comparison_frame, extract_numbers, hallucinated_numbers, number_recall, repeated_trigram_ratio

def test_numbers():
    assert extract_numbers("12.5% in 2025") == {"12.5%", "2025"}
    assert number_recall("value 42", "value 42") == 1.0
    assert number_recall("value", "value 42") == 0.0
    assert hallucinated_numbers("value 99", "value 42") == 1

def test_repetition():
    assert repeated_trigram_ratio("a b c d e") == 0.0
    assert repeated_trigram_ratio("a b c a b c a b c") > 0

def test_comparison():
    frame = comparison_frame({"x":{"samples":2,"rouge1":.4,"rouge2":.2,"rougeL":.3,"bertscore_f1":.8}})
    assert frame.loc[0,"model"] == "x"

def test_error_bands():
    frame = pd.DataFrame({
        "id":["a","b","c","d"], "article":["article 10"]*4, "reference_summary":["summary 10"]*4,
        "prediction":["summary 10","short","wrong 99","summary 10"], "rougeL":[.9,.5,.1,.8],
        "generated_words":[20,2,20,18], "reference_words":[20]*4, "article_words":[100]*4,
        "reference_number_recall":[1,0,0,1], "hallucinated_number_count":[0,0,1,0], "repeated_trigram_ratio":[0,0,.2,0]
    })
    result = build_error_analysis(frame)
    assert set(result.quality_band) <= {"strong","mixed","weak"}
