from players.adapters.scimeto_citations import (
    ScimetoCitationsAdapter,
    _scimeto_to_references,
)


def test_adapter_registers():
    from framework.player_adapter import _ADAPTER_CLASSES
    assert "ScimetoCitationsAdapter" in _ADAPTER_CLASSES


SAMPLE_RESPONSE = {
    "references": [
        {
            "id": "c1",
            "authors": [
                {"surname": "Smith", "given_names": "J A"},
                {"family": "Doe", "given": "R"},
            ],
            "year": "2020",
            "title": "A study of things",
            "venue": "Journal of Things",
            "volume": "12",
            "issue": "3",
            "fpage": "100",
            "lpage": "110",
            "doi": "10.1/ABC",
            "pmid": "99887766",
            "raw_text": "Smith JA, Doe R (2020). A study of things. J Things 12(3):100-110.",
        }
    ]
}


def test_scimeto_to_references():
    refs = _scimeto_to_references(SAMPLE_RESPONSE)
    assert len(refs) == 1
    entry = refs[0]
    assert entry["title"] == "A study of things"
    assert entry["year"] == "2020"
    assert entry["venue"] == "Journal of Things"
    assert entry["volume"] == "12"
    assert entry["issue"] == "3"
    assert entry["fpage"] == "100"
    assert entry["lpage"] == "110"
    assert entry["doi"] == "10.1/abc"
    assert entry["pmid"] == "99887766"
    assert entry["authors"][0]["surname"] == "Smith"
    assert entry["authors"][0]["given_names"] == "J A"
    assert entry["authors"][1]["surname"] == "Doe"
    assert entry["authors"][1]["given_names"] == "R"
    assert entry["reference_id"] == "c1"


def test_scimeto_to_references_accepts_citations_key():
    resp = {"citations": [{"title": "T", "year": 1999}]}
    refs = _scimeto_to_references(resp)
    assert refs[0]["title"] == "T"
    assert refs[0]["year"] == "1999"


def test_scimeto_to_references_tolerates_missing_fields():
    resp = {"references": [{"title": "Bare"}]}
    refs = _scimeto_to_references(resp)
    assert refs[0]["title"] == "Bare"
    assert refs[0]["authors"] == []
    assert refs[0]["year"] is None
    assert refs[0]["doi"] is None
    assert refs[0]["raw_text"] == ""


def test_scimeto_to_references_handles_pages_string():
    resp = {"references": [{"title": "T", "pages": "55-77"}]}
    refs = _scimeto_to_references(resp)
    assert refs[0]["fpage"] == "55"
    assert refs[0]["lpage"] == "77"
