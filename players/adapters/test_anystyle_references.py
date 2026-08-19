from players.adapters.anystyle_references import (
    AnystyleReferencesAdapter,
    _anystyle_json_to_references,
)


def test_adapter_registers():
    from framework.player_adapter import _ADAPTER_CLASSES
    assert "AnystyleReferencesAdapter" in _ADAPTER_CLASSES


def test_anystyle_json_to_references():
    sample = [
        {
            "author": [{"family": "Smith", "given": "J"}],
            "date": ["2020"],
            "title": ["A study"],
            "container-title": ["J Things"],
            "volume": ["12"],
            "issue": ["3"],
            "pages": ["100-110"],
            "doi": ["10.1/a"],
        }
    ]
    refs = _anystyle_json_to_references(sample)
    assert len(refs) == 1
    entry = refs[0]
    assert entry["title"] == "A study"
    assert entry["year"] == "2020"
    assert entry["authors"][0]["surname"] == "Smith"
    assert entry["authors"][0]["given_names"] == "J"
    assert entry["venue"] == "J Things"
    assert entry["volume"] == "12"
    assert entry["issue"] == "3"
    assert entry["fpage"] == "100"
    assert entry["lpage"] == "110"
    assert entry["doi"] == "10.1/a"
    assert "raw_text" in entry


def test_anystyle_json_tolerates_missing_fields():
    refs = _anystyle_json_to_references([{"title": "Bare title", "issued": "1999"}])
    assert refs[0]["title"] == "Bare title"
    assert refs[0]["year"] == "1999"
    assert refs[0]["authors"] == []
    assert refs[0]["venue"] is None
    assert refs[0]["doi"] is None
