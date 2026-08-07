"""Held-out egress gate (2026-08-04).

Sending a held-out task to a third-party provider is IRREVERSIBLE. On 2026-08-04
a `--split revealed` run of a cloud LLM player transmitted 9 held-out real papers
before it was caught — `--split` selects the seed and never filtered visibility.

Two independent guards exist now. The CLI makes `--split revealed` imply
--public-only (test_revealed_split_is_public_only.py). THIS gate lives in the
runner, so it also protects every caller that bypasses the CLI: scripts,
notebooks, the retry queue, any future API.
"""
from __future__ import annotations

import pytest

from framework.runner import (
    _EGRESS_ALLOW_ENV,
    assert_heldout_egress_allowed,
    is_cloud_player,
)

CLOUD = {"player_id": "claude-sonnet-5-sections", "adapter_class": "LlmCliPdfSectionsAdapter"}
CLI_CLOUD = {"player_id": "claude-haiku-4-5-grim", "adapter_class": "SubprocessCliAdapter"}
LOCAL_R = {"player_id": "scrutiny-grim", "adapter_class": "RCliAdapter"}
LOCAL_LIB = {"player_id": "docpluck-sections", "adapter_class": "DocpluckSectionsAdapter"}


def test_llm_pdf_adapter_is_cloud():
    assert is_cloud_player(CLOUD) is True


def test_subprocess_cli_is_cloud():
    """The CLI players are `claude`/`gemini`/`codex`/`opencode` — all remote."""
    assert is_cloud_player(CLI_CLOUD) is True


def test_local_r_and_library_adapters_are_not_cloud():
    assert is_cloud_player(LOCAL_R) is False
    assert is_cloud_player(LOCAL_LIB) is False


def test_unknown_adapter_is_treated_as_local_only_if_prefix_does_not_match():
    """Conservative by construction: a new REMOTE adapter should be named with a
    cloud prefix. Anything genuinely local (a library binding) stays local."""
    assert is_cloud_player({"player_id": "x", "adapter_class": "SomeLocalLibAdapter"}) is False
    assert is_cloud_player({"player_id": "x", "adapter_class": "HttpAdapter"}) is True


def test_gate_blocks_cloud_player_on_held_out(monkeypatch):
    monkeypatch.delenv(_EGRESS_ALLOW_ENV, raising=False)
    with pytest.raises(RuntimeError) as exc:
        assert_heldout_egress_allowed([CLOUD, LOCAL_R], will_play_held_out=True)
    msg = str(exc.value)
    assert "claude-sonnet-5-sections" in msg
    assert _EGRESS_ALLOW_ENV in msg, "the error must name the exact opt-in"
    assert "DATA_HANDLING" in msg
    assert "scrutiny-grim" not in msg, "local players must not be blamed"


def test_gate_allows_public_only_run(monkeypatch):
    """public_only=True means no held-out envelope ever reaches a player."""
    monkeypatch.delenv(_EGRESS_ALLOW_ENV, raising=False)
    assert_heldout_egress_allowed([CLOUD], will_play_held_out=False)


def test_gate_allows_local_players_on_held_out(monkeypatch):
    """Local tools send nothing off-machine, so held-out is fine for them."""
    monkeypatch.delenv(_EGRESS_ALLOW_ENV, raising=False)
    assert_heldout_egress_allowed([LOCAL_R, LOCAL_LIB], will_play_held_out=True)


def test_explicit_env_opt_in_permits_egress(monkeypatch):
    monkeypatch.setenv(_EGRESS_ALLOW_ENV, "1")
    assert_heldout_egress_allowed([CLOUD], will_play_held_out=True)


def test_a_random_env_value_does_not_count_as_consent(monkeypatch):
    monkeypatch.setenv(_EGRESS_ALLOW_ENV, "maybe")
    with pytest.raises(RuntimeError):
        assert_heldout_egress_allowed([CLOUD], will_play_held_out=True)
