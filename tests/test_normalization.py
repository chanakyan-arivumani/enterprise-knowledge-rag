from ingestion.normalization import normalize_text


def test_normalize_line_endings():
    text = "Hello\r\nWorld\rTest"
    assert normalize_text(text) == "Hello\nWorld\nTest"


def test_normalize_trailing_whitespace():
    text = "Hello   \nWorld\t\nTest"
    assert normalize_text(text) == "Hello\nWorld\nTest"


def test_normalize_preserves_blank_lines():
    text = "Hello\n\nWorld"
    assert normalize_text(text) == "Hello\n\nWorld"


def test_normalize_preserves_indentation():
    text = "    pip install something"
    assert normalize_text(text) == "    pip install something"


def test_normalize_removes_bom_and_document_whitespace():
    text = "\ufeff\n\nHello World\n\n"
    assert normalize_text(text) == "Hello World"
