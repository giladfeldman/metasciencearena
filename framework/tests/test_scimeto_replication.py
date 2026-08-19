"""Tests for ScimetoReplicationApiAdapter (HTTP mocked)."""
from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from players.adapters.scimeto_replication import ScimetoReplicationApiAdapter


def _envelope():
    return {
        "task_id": "t",
        "arena_id": "replication-target-lookup-v1",
        "task_set_version": "v1",
        "difficulty": {"tier": 1},
        "input": {"direction": "targets", "doi": "10.1177/1745691616674458", "tier": 1},
    }


def test_adapter_calls_scimeto_and_normalizes(monkeypatch):
    monkeypatch.setenv("SCIMETO_API_URL", "https://example.com/api/v1/replications/extract")
    monkeypatch.setenv("SCIMETO_API_KEY", "test-key")

    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = {
        "direction": "targets",
        "records": [{
            "inputDoi": "10.1177/1745691616674458",
            "originalDoi": "10.1037/0022-3514.54.5.768",
            "replicationDoi": "10.1177/1745691616674458",
            "outcome": "failed",
            "confidence": 0.95,
            "matchMethod": "reference-doi",
            "justificationPhrase": "Registered Replication Report"
        }],
    }
    fake_response.raise_for_status.return_value = None

    adapter = ScimetoReplicationApiAdapter(
        player_id="scimeto-replication-api",
        player_version="0.1.0",
        player_type="platform",
        confidence_strategy="native",
        deterministic=True,
        api_url_env="SCIMETO_API_URL",
        api_key_env="SCIMETO_API_KEY",
    )

    with patch("players.adapters.scimeto_replication.requests.post", return_value=fake_response) as mp:
        out = adapter.play_task(_envelope(), timeout_s=10)
        assert mp.called
        sent_url = mp.call_args.args[0]
        assert "replications/extract" in sent_url
        sent_headers = mp.call_args.kwargs["headers"]
        # The local scimeto API auth contract uses x-internal-api-key (see
        # players/adapters/scimeto_replication.py; commit bd6c908).
        assert sent_headers.get("x-internal-api-key") == "test-key"

    assert out["direction"] == "targets"
    assert out["matches"][0]["original_doi"] == "10.1037/0022-3514.54.5.768"
    assert out["matches"][0]["outcome"] == "failed"
    assert "reference-doi" in out["matches"][0]["provenance"]


def test_adapter_returns_empty_matches_for_unresolved(monkeypatch):
    monkeypatch.setenv("SCIMETO_API_URL", "https://example.com/api/v1/replications/extract")
    monkeypatch.setenv("SCIMETO_API_KEY", "test-key")

    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = {"direction": "targets", "records": []}
    fake_response.raise_for_status.return_value = None

    adapter = ScimetoReplicationApiAdapter(
        player_id="scimeto-replication-api",
        player_version="0.1.0",
        player_type="platform",
        confidence_strategy="native",
        deterministic=True,
        api_url_env="SCIMETO_API_URL",
        api_key_env="SCIMETO_API_KEY",
    )

    with patch("players.adapters.scimeto_replication.requests.post", return_value=fake_response):
        out = adapter.play_task(_envelope(), timeout_s=10)
    assert out["matches"] == []
