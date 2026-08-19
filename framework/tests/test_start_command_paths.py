"""`start_command` must resolve through $VIBE_ROOT, never a hardcoded user path.

Regression for a defect found 2026-08-08 while verifying the grant-application
materials: `players/registry.yaml` started the `escimate` platform player from

    C:/Users/filin/Dropbox/Vibe/MetaScienceTools/ESCIcheckapp/start_app.bat

which stopped existing on 2026-08-03, when the Vibe portfolio was moved out of
Dropbox (Dropbox syncing a live `.git` corrupts repos). The real file now lives
at `$VIBE_ROOT/MetaScienceTools/ESCIcheckapp/start_app.bat`.

Two things were wrong and both are pinned here:

1. The path was hardcoded to one machine's home directory, which the
   portfolio-wide rule in `Vibe/CLAUDE.md` forbids outright.
2. `HttpAdapter.prepare()` handed the dead path straight to `subprocess.Popen`.
   On Windows `cmd /c missing.bat` *succeeds* — it exits 1 and Popen is happy —
   so the failure surfaced 60 seconds later as "ESCImate did not become ready",
   which reads like a slow server rather than a path that cannot exist. A wrong
   cause is worse than no message; it sends the next reader to the wrong file.
"""
import os
from pathlib import Path

import pytest
import yaml

from framework.player_adapter import HttpAdapter, resolve_start_command

REPO_ROOT = Path(__file__).resolve().parents[2]


def _registry_players():
    return yaml.safe_load((REPO_ROOT / "players" / "registry.yaml").read_text(encoding="utf-8"))


def test_no_registry_start_command_hardcodes_a_user_home():
    """The literal defect: an absolute per-machine path in the committed registry."""
    offenders = []
    for player in _registry_players():
        for token in player.get("start_command") or []:
            lowered = token.lower()
            if "/users/" in lowered or "\\users\\" in lowered or "dropbox" in lowered:
                offenders.append((player["player_id"], token))
    assert offenders == [], (
        "start_command must resolve through ${VIBE_ROOT}, not a hardcoded home "
        f"directory. Offending entries: {offenders}"
    )


def test_resolve_start_command_expands_vibe_root(monkeypatch, tmp_path):
    monkeypatch.setenv("VIBE_ROOT", str(tmp_path))
    resolved = resolve_start_command(["cmd", "/c", "${VIBE_ROOT}/Tools/start_app.bat"])
    assert resolved == ["cmd", "/c", str(tmp_path / "Tools" / "start_app.bat").replace(os.sep, "/")]


def test_resolve_start_command_defaults_vibe_root_to_home_vibe(monkeypatch, tmp_path):
    """Matching the portfolio rule's documented order: $VIBE_ROOT, else $HOME/Vibe."""
    monkeypatch.delenv("VIBE_ROOT", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    resolved = resolve_start_command(["${VIBE_ROOT}/x.bat"])
    assert resolved == [str(tmp_path / "Vibe" / "x.bat").replace(os.sep, "/")]


def test_prepare_fails_loudly_when_the_script_is_missing(monkeypatch, tmp_path):
    """A missing launcher must name the missing path, not time out for 60s."""
    monkeypatch.setenv("VIBE_ROOT", str(tmp_path))
    adapter = HttpAdapter(
        player_id="escimate",
        player_type="platform",
        player_version="0.0.0",
        confidence_strategy="implicit-1.0",
        deterministic=True,
        endpoint="http://127.0.0.1:59999/api/v1/process-text",
        start_command=["cmd", "/c", "${VIBE_ROOT}/nope/start_app.bat"],
    )
    with pytest.raises(FileNotFoundError, match="nope/start_app.bat"):
        adapter.prepare()


def test_escimate_launcher_actually_exists_where_the_registry_points():
    """The registry's path must point at a real file on this machine.

    Skipped rather than failed when the portfolio root is absent, so a CI box or
    a fresh clone does not fail on a tool it was never expected to have.
    """
    players = {p["player_id"]: p for p in _registry_players()}
    escimate = players.get("escimate")
    if escimate is None or not escimate.get("start_command"):
        pytest.skip("escimate has no start_command")

    vibe_root = Path(os.environ.get("VIBE_ROOT") or Path.home() / "Vibe")
    if not vibe_root.is_dir():
        pytest.skip(f"portfolio root {vibe_root} not present on this machine")

    script = Path(resolve_start_command(escimate["start_command"])[-1])
    assert script.exists(), (
        f"registry points escimate at {script}, which does not exist. "
        "The launcher moved; update players/registry.yaml."
    )
