from players.adapters.cermine_references import (
    CermineReferencesAdapter,
    _nlm_to_references,
)


def test_adapter_registers():
    from framework.player_adapter import _ADAPTER_CLASSES
    assert "CermineReferencesAdapter" in _ADAPTER_CLASSES


SAMPLE_NLM = """<ref-list>
  <ref id="ref1">
    <mixed-citation>
      <string-name><surname>Smith</surname><given-names>J A</given-names></string-name>
      <string-name><surname>Doe</surname><given-names>R</given-names></string-name>
      <article-title>A study of things</article-title>
      <source>Journal of Things</source>
      <year>2020</year>
      <volume>12</volume>
      <issue>3</issue>
      <fpage>100</fpage>
      <lpage>110</lpage>
      <pub-id pub-id-type="doi">10.1/abc</pub-id>
    </mixed-citation>
  </ref>
  <ref id="ref2">
    <mixed-citation>
      <string-name><surname>Lone</surname><given-names>K</given-names></string-name>
      <article-title>Bare</article-title>
      <year>1999</year>
    </mixed-citation>
  </ref>
</ref-list>"""


def test_nlm_to_references():
    refs = _nlm_to_references(SAMPLE_NLM)
    assert len(refs) == 2
    first = refs[0]
    assert first["title"] == "A study of things"
    assert first["year"] == "2020"
    assert first["venue"] == "Journal of Things"
    assert first["volume"] == "12"
    assert first["issue"] == "3"
    assert first["fpage"] == "100"
    assert first["lpage"] == "110"
    assert first["doi"] == "10.1/abc"
    assert first["authors"][0]["surname"] == "Smith"
    assert first["authors"][0]["given_names"] == "J A"
    assert first["authors"][1]["surname"] == "Doe"
    assert first["reference_id"] == "ref1"


def test_nlm_to_references_tolerates_missing_fields():
    refs = _nlm_to_references(SAMPLE_NLM)
    second = refs[1]
    assert second["title"] == "Bare"
    assert second["year"] == "1999"
    assert second["venue"] is None
    assert second["volume"] is None
    assert second["doi"] is None
    assert "raw_text" in second


def test_nlm_namespace_agnostic():
    namespaced = SAMPLE_NLM.replace(
        "<ref-list>", '<ref-list xmlns="http://jats.nlm.nih.gov">')
    refs = _nlm_to_references(namespaced)
    assert refs[0]["title"] == "A study of things"
