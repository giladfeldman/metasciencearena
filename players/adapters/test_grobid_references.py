from players.adapters.grobid_references import GrobidReferencesAdapter


def test_adapter_registers_and_constructs():
    from framework.player_adapter import _ADAPTER_CLASSES
    assert "GrobidReferencesAdapter" in _ADAPTER_CLASSES


def test_tei_to_references_parses_biblstruct():
    adapter = GrobidReferencesAdapter(
        player_id="grobid-references", player_version="0.8.1",
        player_type="tool", confidence_strategy="implicit-1.0",
        deterministic=True, endpoint="http://localhost:8070")
    tei = '''<TEI xmlns="http://www.tei-c.org/ns/1.0"><text><back><listBibl>
      <biblStruct><analytic><title>A study</title>
      <author><persName><surname>Smith</surname></persName></author></analytic>
      <monogr><title>J Things</title><imprint><date when="2020"/></imprint></monogr>
      </biblStruct></listBibl></back></text></TEI>'''
    refs = adapter._tei_to_references(tei)
    assert refs[0]["title"] == "A study"
    assert refs[0]["year"] == "2020"
