from players.adapters.llm_pdf_citations import (
    LlmPdfCitationsAdapter,
    _coerce_citations,
)


def test_adapter_registers():
    from framework.player_adapter import _ADAPTER_CLASSES
    assert "LlmPdfCitationsAdapter" in _ADAPTER_CLASSES


def test_coerce_citations_strips_json_fence():
    fenced = ('```json\n{"markers": [], "consistency": '
              '{"orphan_markers": [], "uncited_reference_ids": [], '
              '"duplicate_reference_groups": []}}\n```')
    out = _coerce_citations(fenced)
    assert out == {
        "markers": [],
        "consistency": {
            "orphan_markers": [],
            "uncited_reference_ids": [],
            "duplicate_reference_groups": [],
        },
    }


def test_coerce_citations_parses_plain_json():
    raw = ('{"markers": [{"marker_text": "[1]", "char_start": 10, '
           '"char_end": 13, "reference_id": "r1"}], '
           '"consistency": {"orphan_markers": ["[9]"], '
           '"uncited_reference_ids": ["r5"], '
           '"duplicate_reference_groups": [["r2", "r3"]]}}')
    out = _coerce_citations(raw)
    assert out["markers"][0]["marker_text"] == "[1]"
    assert out["markers"][0]["char_start"] == 10
    assert out["markers"][0]["reference_id"] == "r1"
    assert out["consistency"]["orphan_markers"] == ["[9]"]
    assert out["consistency"]["duplicate_reference_groups"] == [["r2", "r3"]]


def test_coerce_citations_defaults_missing_consistency():
    out = _coerce_citations('{"markers": [{"marker_text": "[1]"}]}')
    assert out["markers"][0]["marker_text"] == "[1]"
    assert out["consistency"] == {
        "orphan_markers": [],
        "uncited_reference_ids": [],
        "duplicate_reference_groups": [],
    }


def test_coerce_citations_defaults_missing_markers():
    out = _coerce_citations('{"consistency": {"orphan_markers": ["x"], '
                            '"uncited_reference_ids": [], '
                            '"duplicate_reference_groups": []}}')
    assert out["markers"] == []
    assert out["consistency"]["orphan_markers"] == ["x"]


def test_coerce_citations_tolerates_partial_marker_fields():
    out = _coerce_citations('{"markers": [{"marker_text": "[2]"}]}')
    marker = out["markers"][0]
    assert marker["char_start"] is None
    assert marker["char_end"] is None
    assert marker["reference_id"] is None


def test_coerce_citations_handles_garbage():
    out = _coerce_citations("sorry, I cannot do that")
    assert out["markers"] == []
    assert out["consistency"]["orphan_markers"] == []


def test_coerce_citations_finds_embedded_object():
    chatty = ('Here is the result:\n{"markers": [{"marker_text": "[3]"}]}\n'
              'Done.')
    out = _coerce_citations(chatty)
    assert out["markers"][0]["marker_text"] == "[3]"


def test_prompt_template_exists():
    from pathlib import Path
    tpl = (Path(__file__).resolve().parents[1] / "prompts"
           / "pdf_citation_matching.txt")
    assert tpl.is_file()
    text = tpl.read_text(encoding="utf-8")
    assert "{{PDF_PATH}}" in text
    assert "markers" in text
    assert "consistency" in text
