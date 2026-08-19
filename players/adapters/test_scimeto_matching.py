from players.adapters.scimeto_matching import (
    ScimetoMatchingAdapter,
    _scimeto_to_linkage,
)


def test_adapter_registers():
    from framework.player_adapter import _ADAPTER_CLASSES
    assert "ScimetoMatchingAdapter" in _ADAPTER_CLASSES


SAMPLE_RESPONSE = {
    "in_text_citations": [
        {
            "marker_text": "[1]",
            "char_start": 120,
            "char_end": 123,
            "reference_id": "r1",
        },
        {
            "marker_text": "[9]",
            "char_start": 400,
            "char_end": 403,
            "reference_id": None,
        },
    ],
    "consistency": {
        "orphan_markers": ["[9]"],
        "uncited_reference_ids": ["r5"],
        "duplicate_reference_groups": [["r2", "r3"]],
    },
}


def test_scimeto_to_linkage():
    out = _scimeto_to_linkage(SAMPLE_RESPONSE)
    assert "markers" in out and "consistency" in out
    markers = out["markers"]
    assert markers[0]["marker_text"] == "[1]"
    assert markers[0]["char_start"] == 120
    assert markers[0]["reference_id"] == "r1"
    cons = out["consistency"]
    for key in ("orphan_markers", "uncited_reference_ids",
                "duplicate_reference_groups"):
        assert key in cons
    assert cons["orphan_markers"] == ["[9]"]
    assert cons["uncited_reference_ids"] == ["r5"]
    assert cons["duplicate_reference_groups"] == [["r2", "r3"]]


def test_scimeto_to_linkage_unwraps_data_envelope():
    resp = {"success": True, "data": SAMPLE_RESPONSE}
    out = _scimeto_to_linkage(resp)
    assert out["markers"][0]["reference_id"] == "r1"


def test_scimeto_to_linkage_accepts_markers_alias():
    resp = {"markers": [{"marker_text": "(Smith, 2020)", "reference_id": "r7"}]}
    out = _scimeto_to_linkage(resp)
    assert out["markers"][0]["marker_text"] == "(Smith, 2020)"
    assert out["markers"][0]["reference_id"] == "r7"


def test_scimeto_to_linkage_tolerates_missing_fields():
    resp = {"in_text_citations": [{"marker_text": "[2]"}]}
    out = _scimeto_to_linkage(resp)
    marker = out["markers"][0]
    assert marker["marker_text"] == "[2]"
    assert marker["char_start"] == -1
    assert marker["char_end"] == -1
    assert marker["reference_id"] is None
    cons = out["consistency"]
    assert cons["orphan_markers"] == []
    assert cons["uncited_reference_ids"] == []
    assert cons["duplicate_reference_groups"] == []


def test_scimeto_to_linkage_handles_garbage():
    out = _scimeto_to_linkage("not a dict")
    assert out["markers"] == []
    assert out["consistency"]["orphan_markers"] == []
