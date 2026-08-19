from players.adapters.llm_pdf_references import (
    LlmPdfReferencesAdapter,
    _coerce_references,
)


def test_adapter_registers():
    from framework.player_adapter import _ADAPTER_CLASSES
    assert "LlmPdfReferencesAdapter" in _ADAPTER_CLASSES


def test_coerce_references_strips_json_fence():
    fenced = '```json\n{"references": []}\n```'
    assert _coerce_references(fenced) == {"references": []}


def test_coerce_references_parses_plain_json():
    out = _coerce_references('{"references": [{"raw_text": "x", "title": "T"}]}')
    assert out["references"][0]["title"] == "T"
    assert out["references"][0]["raw_text"] == "x"


def test_coerce_references_normalizes_entry_fields():
    raw = ('{"references": [{"title": "A study", "year": 2020, '
           '"authors": [{"surname": "Smith", "given_names": "J"}], '
           '"doi": "10.1/ABC"}]}')
    out = _coerce_references(raw)
    entry = out["references"][0]
    assert entry["title"] == "A study"
    assert entry["year"] == "2020"
    assert entry["doi"] == "10.1/abc"
    assert entry["authors"][0]["surname"] == "Smith"
    assert entry["raw_text"] == ""
    # all schema keys present
    for key in ("reference_id", "venue", "volume", "issue",
                "fpage", "lpage", "pmid"):
        assert key in entry


def test_coerce_references_handles_garbage():
    assert _coerce_references("sorry, I cannot do that") == {"references": []}


def test_coerce_references_finds_embedded_object():
    chatty = 'Here is the result:\n{"references": [{"raw_text": "r"}]}\nDone.'
    out = _coerce_references(chatty)
    assert out["references"][0]["raw_text"] == "r"


def test_prompt_template_exists():
    from pathlib import Path
    tpl = (Path(__file__).resolve().parents[1] / "prompts"
           / "pdf_reference_parsing.txt")
    assert tpl.is_file()
    text = tpl.read_text(encoding="utf-8")
    assert "{{PDF_PATH}}" in text
    assert "references" in text
