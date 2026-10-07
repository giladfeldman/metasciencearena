from players.adapters.grobid_tables import GrobidTablesAdapter

_extract_tables = GrobidTablesAdapter._extract_tables


def test_adapter_registers():
    from framework.player_adapter import _ADAPTER_CLASSES
    assert "GrobidTablesAdapter" in _ADAPTER_CLASSES


SAMPLE_TEI = '''<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <text><body>
    <figure type="table">
      <head>Table 1</head>
      <figDesc>Demographics</figDesc>
      <table>
        <row><cell role="head">Group</cell><cell role="head">N</cell></row>
        <row><cell>Treatment</cell><cell>42</cell></row>
        <row><cell>Control</cell><cell>40</cell></row>
      </table>
    </figure>
  </body></text>
</TEI>'''


def test_extract_tables_parses_a_table():
    tables = _extract_tables(SAMPLE_TEI)
    assert len(tables) == 1
    t = tables[0]
    assert t["label"] == "Table 1"
    assert t["caption"] == "Demographics"
    assert t["n_rows"] == 3
    assert t["n_cols"] == 2
    assert t["header_rows"] == 1
    assert any(c["is_header"] for c in t["cells"])


def test_extract_tables_malformed_xml_returns_empty():
    # DR-0014: a malformed GROBID response must degrade to "no tables", not raise
    # an uncaught XMLSyntaxError that crashes the task.
    assert _extract_tables("not xml at all <<<") == []
    assert _extract_tables("<TEI><unclosed>") == []
    assert _extract_tables("") == []


def test_extract_tables_no_tables_returns_empty():
    no_tables = '<TEI xmlns="http://www.tei-c.org/ns/1.0"><text><body><p>No tables here.</p></body></text></TEI>'
    assert _extract_tables(no_tables) == []
