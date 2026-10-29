"""The GROBID liveness probe must tolerate a loaded machine (debt d-28dd52).

2026-10-24: /api/isalive took 2.7 s while other extractors ran, past the 3 s probe
timeout, and both GROBID section runs aborted at task 0 with "server not reachable"
although the server was up.
"""
import pytest

from players.adapters import grobid_citations, grobid_references, grobid_sections, grobid_tables, grobid_text


@pytest.mark.parametrize("mod", [grobid_citations, grobid_references, grobid_sections, grobid_tables, grobid_text])
def test_isalive_probe_allows_a_slow_answer(mod, monkeypatch):
    seen = {}

    class _Ok:
        status_code = 200

    def fake_get(url, timeout=None, **kw):
        seen["timeout"] = timeout
        return _Ok()

    monkeypatch.setattr(mod.requests, "get", fake_get)
    # grobid_sections reads its label map from a private arena before probing; the
    # public mirror does not carry that file, so stub it or the probe is never reached.
    if hasattr(mod, "_load_grobid_label_map"):
        monkeypatch.setattr(mod, "_load_grobid_label_map", lambda: {})
    cls = next(v for k, v in vars(mod).items() if k.startswith("Grobid") and k.endswith("Adapter"))
    adapter = cls.__new__(cls)
    adapter.endpoint = "http://localhost:1"
    for attr in ("_label_map",):
        setattr(adapter, attr, None)
    try:
        adapter.prepare()
    except Exception:
        pass
    assert seen.get("timeout", 0) >= 15
