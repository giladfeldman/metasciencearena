"""The stats-from-pdf pipelines must never send a PDF off this machine."""
from __future__ import annotations

from pathlib import Path

import pytest

from players.adapters.stats_pipeline import StatsPipelineAdapter

R = Path(__file__).resolve().parents[1] / "stats_pipeline.R"


def _make(**kw):
    return StatsPipelineAdapter(player_id="x", player_version="x", player_type="tool",
                                confidence_strategy="implicit-1.0", deterministic=True, **kw)


@pytest.mark.parametrize("field,url", [("grobid_endpoint", "https://grobid.hti.ieis.tue.nl"),
                                       ("grobid_endpoint", "http://127.0.0.1.attacker.example"),
                                       ("docpluck_url", "https://docpluck.app")])
def test_remote_service_urls_are_refused(field, url):
    with pytest.raises(ValueError, match="LOCAL services only"):
        _make(extractor="grobid", checker="statcheck", **{field: url})


def test_loopback_urls_are_accepted():
    _make(extractor="grobid", checker="statcheck", grobid_endpoint="http://localhost:8072",
          docpluck_url="http://127.0.0.1:6117")


def test_metacheck_route_never_uses_its_hosted_defaults():
    src = R.read_text(encoding="utf-8")
    assert "api_url = req$grobid_url" in src          # the local GROBID, not metacheck's default server
    assert 'stop("metacheck on bibr needs the local bibr setup' in src   # never the hosted bibr
