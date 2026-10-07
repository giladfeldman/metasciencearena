"""Runtime version detection for R-backed reference tools (F7, cycle 8, 2026-08-04).

Before this, `RCliAdapter.resolved_tool_version()` inherited the base-class default
of None, so `framework audit --versions` reported drift for the 12 Python-backed
tools and NOTHING for any R tool — scrutiny, metacheck, oddpub, rtransparent,
statcheck, metafor, effectsize, rsprite2, zcurve. Those are precisely the
deterministic tools whose ~1.00 score cross-validates an arena's gold, so an
undetected R upgrade would silently re-rank published history. Cycle 6 had to
reconcile their labels by hand, which is the symptom this closes.
"""
from __future__ import annotations

import shutil

import pytest

from framework.player_adapter import build_adapter, resolve_rscript_binary
from players.adapters._tool_version import r_adapter_package, r_package_version

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
ADAPTERS = REPO_ROOT / "players" / "adapters"


def _rscript_available() -> bool:
    return shutil.which(resolve_rscript_binary(None)) is not None


# --------------------------------------------------------------------------- #
# Package derivation — pure parsing, no R required.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("script", "expected"), [
    ("grim_scrutiny.R", "scrutiny"),
    ("metacheck.R", "metacheck"),
    ("sprite_rsprite2.R", "rsprite2"),
    ("zcurve_tool.R", "zcurve"),
    ("statcheck.R", "statcheck"),
    ("metafor_pubbias.R", "metafor"),
    ("effectsize_convert.R", "effectsize"),
    ("oddpub.R", "oddpub"),
    ("rtransparent.R", "rtransparent"),
])
def test_r_adapter_package_derived_from_script(script, expected):
    """The package is READ FROM the script, not a hardcoded table, so a new R
    adapter gets version detection for free and a renamed dependency cannot
    silently desync."""
    assert r_adapter_package(ADAPTERS / script) == expected


def test_r_adapter_package_skips_jsonlite_plumbing():
    """jsonlite is I/O plumbing every adapter loads — never the tool being versioned."""
    assert r_adapter_package(ADAPTERS / "grim_scrutiny.R") != "jsonlite"


def test_r_adapter_package_none_for_base_r_adapter():
    """pcurve.R is a pure base-R implementation with no package to version."""
    assert r_adapter_package(ADAPTERS / "pcurve.R") is None


def test_r_adapter_package_none_for_missing_file():
    assert r_adapter_package(ADAPTERS / "does_not_exist.R") is None


def test_r_package_version_none_for_empty_package():
    assert r_package_version("") is None


def test_r_package_version_none_for_missing_package():
    """Best-effort contract: an uninstalled package must return None, never raise."""
    assert r_package_version("definitely_not_a_real_r_package_xyz") is None


# --------------------------------------------------------------------------- #
# End-to-end against the real R install.
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _rscript_available(),
                    reason="Rscript unavailable (set RSCRIPT_BINARY)")
def test_rcli_adapter_resolves_real_r_package_version():
    """The scrutiny-grim player must report a concrete scrutiny version.

    This is the F7 regression: it returned None for every R tool before the fix.
    """
    adapter = build_adapter({
        "player_id": "scrutiny-grim", "player_type": "tool",
        "player_version": "scrutiny-grim", "adapter_class": "RCliAdapter",
        "confidence_strategy": "implicit-1.0", "deterministic": True,
        "r_script": str(ADAPTERS / "grim_scrutiny.R"),
    })
    resolved = adapter.resolved_tool_version()
    assert resolved is not None, "R tool version detection returned None (F7 regression)"
    assert resolved.startswith("scrutiny-")
    # A real dotted version, not a placeholder.
    assert any(ch.isdigit() for ch in resolved.split("scrutiny-", 1)[1])


@pytest.mark.skipif(not _rscript_available(),
                    reason="Rscript unavailable (set RSCRIPT_BINARY)")
def test_base_r_adapter_still_resolves_none():
    """A base-R adapter has no package version — None is correct, not a failure."""
    adapter = build_adapter({
        "player_id": "pcurve-tool", "player_type": "tool",
        "player_version": "pcurve", "adapter_class": "RCliAdapter",
        "confidence_strategy": "implicit-1.0", "deterministic": True,
        "r_script": str(ADAPTERS / "pcurve.R"),
    })
    assert adapter.resolved_tool_version() is None
