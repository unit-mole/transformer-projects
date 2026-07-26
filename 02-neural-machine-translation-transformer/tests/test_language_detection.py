from src.language_detection import detect_language


def test_detects_english():
    assert detect_language("The quality report is ready.").language == "english"


def test_detects_hindi():
    assert detect_language("गुणवत्ता रिपोर्ट तैयार है।").language == "hindi"


def test_detects_mixed():
    assert detect_language("Hello नमस्ते").language == "mixed"


def test_detects_uncertain():
    assert detect_language("12345 !").language == "uncertain"
