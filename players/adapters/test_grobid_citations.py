from players.adapters.grobid_citations import (
    GrobidCitationsAdapter,
    _tei_to_linkage,
)


def test_adapter_registers():
    from framework.player_adapter import _ADAPTER_CLASSES
    assert "GrobidCitationsAdapter" in _ADAPTER_CLASSES


SAMPLE_TEI = '''<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <text>
    <body>
      <p>Earlier work <ref type="bibr" target="#b1">1</ref> established the
      result, and a later study <ref type="bibr" target="#b99">7</ref>
      challenged it.</p>
    </body>
    <back>
      <div type="references">
        <listBibl>
          <biblStruct xml:id="b1">
            <analytic><title>A study</title></analytic>
            <monogr><idno type="DOI">10.1/ABC</idno></monogr>
          </biblStruct>
          <biblStruct xml:id="b2">
            <analytic><title>An uncited work</title></analytic>
          </biblStruct>
        </listBibl>
      </div>
    </back>
  </text>
</TEI>'''


def test_tei_to_linkage():
    out = _tei_to_linkage(SAMPLE_TEI)
    assert "markers" in out and "consistency" in out
    markers = out["markers"]
    # the b1 marker resolves; the b99 marker does not
    assert markers[0]["marker_text"] == "1"
    assert markers[0]["reference_id"] == "b1"
    cons = out["consistency"]
    for key in ("orphan_markers", "uncited_reference_ids",
                "duplicate_reference_groups"):
        assert key in cons


def test_tei_to_linkage_orphan_and_uncited():
    out = _tei_to_linkage(SAMPLE_TEI)
    cons = out["consistency"]
    # marker targeting #b99 resolves to no biblStruct -> orphan
    assert "7" in cons["orphan_markers"]
    # b2 is never targeted by any marker -> uncited
    assert "b2" in cons["uncited_reference_ids"]
    # the unresolved marker carries reference_id None
    assert out["markers"][1]["reference_id"] is None


def test_tei_to_linkage_duplicate_groups_by_doi():
    tei = '''<TEI xmlns="http://www.tei-c.org/ns/1.0"><text>
      <body><p><ref type="bibr" target="#b1">1</ref></p></body>
      <back><listBibl>
        <biblStruct xml:id="b1"><analytic><title>X</title>
          <idno type="DOI">10.1/SAME</idno></analytic></biblStruct>
        <biblStruct xml:id="b2"><analytic><title>Y</title>
          <idno type="DOI">10.1/same</idno></analytic></biblStruct>
      </listBibl></back></text></TEI>'''
    out = _tei_to_linkage(tei)
    groups = out["consistency"]["duplicate_reference_groups"]
    assert any(set(g) == {"b1", "b2"} for g in groups)


def test_tei_to_linkage_handles_garbage():
    assert _tei_to_linkage("not xml at all") == {
        "markers": [],
        "consistency": {
            "orphan_markers": [],
            "uncited_reference_ids": [],
            "duplicate_reference_groups": [],
        },
    }
