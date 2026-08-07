"""Tests for RCliAdapter via a fake R script (does not require R installed for unit tests)."""
import shutil

from framework.player_adapter import build_adapter, resolve_rscript_binary
from framework.runner import _adapter_command

R_AVAILABLE = shutil.which("Rscript") is not None

_ENTRY = {
    "player_id": "rcli-stub", "player_type": "tool", "player_version": "1",
    "adapter_class": "RCliAdapter", "confidence_strategy": "implicit-1.0",
    "deterministic": True, "r_script": "/nonexistent.R",
}


def test_build_rcli_adapter_constructs():
    """The adapter builds and carries the script path through.

    Regression: this test previously ended on a bare `build_adapter(...)` call with
    NO assertion, so it could never fail — a false green (see the portfolio
    "No pretending" rule). Assert the constructed object.
    """
    adapter = build_adapter(dict(_ENTRY))
    assert adapter is not None
    # Path() normalizes separators per-platform ("\nonexistent.R" on Windows).
    assert adapter.r_script.name == "nonexistent.R"
    assert adapter.rscript_binary


def test_resolve_rscript_binary_prefers_explicit_over_env(monkeypatch):
    monkeypatch.setenv("RSCRIPT_BINARY", "C:/env/Rscript.exe")
    assert resolve_rscript_binary("C:/explicit/Rscript.exe") == "C:/explicit/Rscript.exe"


def test_resolve_rscript_binary_uses_env_when_no_explicit(monkeypatch):
    """RSCRIPT_BINARY makes the R toolchain reproducible when R is not on PATH.

    Regression (2026-08-04, cycle 7): R 4.4.0 is installed on this dev box but NOT on
    PATH, so the hardcoded bare `Rscript` raised FileNotFoundError and EVERY R
    reference tool recorded as an errored task — scrutiny-grim scored 0 instead of
    cross-validating the arena's gold at ~1.00. Silent, because the R smoke tests all
    `skipif(which("Rscript") is None)`.
    """
    monkeypatch.setenv("RSCRIPT_BINARY", "C:/env/Rscript.exe")
    assert resolve_rscript_binary(None) == "C:/env/Rscript.exe"


def test_resolve_rscript_binary_falls_back_to_path(monkeypatch):
    monkeypatch.delenv("RSCRIPT_BINARY", raising=False)
    assert resolve_rscript_binary(None) == "Rscript"


def test_resolve_rscript_binary_ignores_blank_env(monkeypatch):
    monkeypatch.setenv("RSCRIPT_BINARY", "   ")
    assert resolve_rscript_binary(None) == "Rscript"


def test_rcli_adapter_honours_env_binary(monkeypatch):
    monkeypatch.setenv("RSCRIPT_BINARY", "C:/env/Rscript.exe")
    adapter = build_adapter(dict(_ENTRY))
    assert adapter.rscript_binary == "C:/env/Rscript.exe"


def test_adapter_command_card_reports_resolved_binary(monkeypatch):
    """The run record's `command` must name the binary actually invoked.

    Provenance defect otherwise: the card said "Rscript ..." while the process ran
    a fully-qualified path (or failed to run at all).
    """
    monkeypatch.setenv("RSCRIPT_BINARY", "C:/env/Rscript.exe")
    card = _adapter_command(dict(_ENTRY))
    assert "C:/env/Rscript.exe" in card
    assert "rcli-stub" in card
