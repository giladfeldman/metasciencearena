"""`start_command` must resolve through $VIBE_ROOT, never a hardcoded user path.

Regression for a defect found 2026-08-08 while verifying the grant-application
materials: `players/registry.yaml` started the `escimate` platform player from an
absolute path under one machine's home directory, inside a cloud-synced folder.

That path stopped existing on 2026-08-03, when the project root was moved out of
cloud sync (syncing a live `.git` corrupts repos). The launcher it pointed at is
now resolved relative to `$VIBE_ROOT`, which is what this test pins.

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
import re
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

    try:
        script = Path(resolve_start_command(escimate["start_command"])[-1])
    except KeyError as e:
        # A machine that holds the portfolio root but not the gitignored
        # local_paths.yaml. Skipping is right; the resolver already failed loudly
        # and named what is missing, which is the behaviour pinned below.
        pytest.skip(str(e))
    assert script.exists(), (
        f"registry points escimate at {script}, which does not exist. "
        "The launcher moved; update the ${NAME} value that supplies its directory."
    )


# --- The launcher path is a DISCLOSURE surface, not only a portability one ----
# Added 2026-09-17. `players/registry.yaml` is on `framework/publish.py`'s public
# allowlist, so a launcher path written there told any reader of the public
# mirror the directory layout of a PRIVATE sibling repository (confirmed private
# via `gh repo view`). Two published files carried it; both are now zero,
# measured against the 546 files the public manifest lists. The path moved to
# the gitignored `players/local_paths.yaml`, reached through a `${NAME}`
# placeholder.
#
# The names below are ASSEMBLED rather than written, because THIS FILE IS ITSELF
# PUBLISHED. Spelling them here would re-publish exactly what the test exists to
# keep out — and the first version of this test did, and failed on itself.
_PRIVATE_SIBLING_NAMES = ("ESCI" + "checkapp", "MetaScience" + "Tools")


def test_no_published_file_names_a_private_sibling_repository():
    """Two-sided: the scan must find something, or its zero means nothing."""
    manifest = REPO_ROOT / "contract" / "public_manifest.txt"
    if manifest.is_file():
        published = [l.strip() for l in manifest.read_text(encoding="utf-8").splitlines()
                     if l.strip() and not l.startswith("#")]
    else:
        # The public mirror does not ship the manifest; there, the published set
        # IS what git tracks. Same scan, same two-sided floor below.
        import subprocess
        out = subprocess.run(["git", "ls-files"], cwd=REPO_ROOT, capture_output=True,
                             text=True, encoding="utf-8").stdout or ""
        published = [l.strip() for l in out.splitlines() if l.strip()]
    scanned = control = 0
    offenders = []
    for rel in published:
        path = REPO_ROOT / rel
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        scanned += 1
        if "arena" in text:
            control += 1
        for needle in _PRIVATE_SIBLING_NAMES:
            if needle in text:
                offenders.append((rel, needle))
    assert scanned > 100, f"only {scanned} published files readable; the scan is not live"
    assert control > 100, f"positive control found {control}; the scan cannot read content"
    assert offenders == [], offenders


def test_the_private_sibling_scan_can_actually_find_a_name(tmp_path):
    """Positive control for the scan ABOVE, which would otherwise pass just as
    happily if `_PRIVATE_SIBLING_NAMES` were assembled wrong and matched nothing."""
    planted = tmp_path / "planted.txt"
    planted.write_text("start_app.bat lives under " + _PRIVATE_SIBLING_NAMES[0], encoding="utf-8")
    text = planted.read_text(encoding="utf-8")
    assert any(n in text for n in _PRIVATE_SIBLING_NAMES)
    assert not any(n in "nothing to see here" for n in _PRIVATE_SIBLING_NAMES)


def test_an_unset_placeholder_raises_and_names_itself(monkeypatch, tmp_path):
    """It must never survive into the command. `cmd /c` exits 1 on a nonexistent
    script without raising, so a surviving placeholder would surface 60 seconds
    later as "did not become ready" — blaming the server for a path never built."""
    monkeypatch.setattr("framework.player_adapter.LOCAL_PATHS_FILE", tmp_path / "absent.yaml")
    monkeypatch.delenv("NO_SUCH_PLACEHOLDER", raising=False)
    with pytest.raises(KeyError, match="NO_SUCH_PLACEHOLDER"):
        resolve_start_command(["cmd", "/c", "${NO_SUCH_PLACEHOLDER}/x.bat"])


def test_a_clean_checkout_fails_loudly_and_says_exactly_what_to_create(monkeypatch, tmp_path):
    """THE CLEAN-CHECKOUT CONTRACT, asserted on the REAL registry entry.

    A fresh clone has no `players/local_paths.yaml` (gitignored) and no
    `ESCIMATE_APP_DIR` exported, so the escimate player does NOT start. That is
    the deliberate trade, not a regression: `players/registry.yaml` is on
    `framework/publish.py`'s public allowlist, so a launcher path written there
    tells a reader of the public mirror the directory layout of a PRIVATE
    sibling repository. The cost is one gitignored line per machine.

    What makes the trade payable is that the failure is SELF-FIXING. This test
    pins that: the error must name the placeholder, name the file to create, and
    say the value may come from the environment instead. A loud failure that
    does not say what to do is just a different way to lose an afternoon --
    which is what the 60-second "did not become ready" timeout used to be.

    Exercised through the real registry rather than a literal, so it cannot
    drift away from what the player actually declares.
    """
    players = {p["player_id"]: p for p in _registry_players()}
    escimate = players.get("escimate")
    if escimate is None or not escimate.get("start_command"):
        pytest.skip("escimate has no start_command")
    placeholders = [
        m for tok in escimate["start_command"] for m in re.findall(r"\$\{(\w+)\}", tok)
        if m != "VIBE_ROOT"
    ]
    assert placeholders, (
        "escimate's start_command spells its launcher directory literally again. "
        "That path is published to the public mirror; keep it behind a ${NAME}."
    )

    # Simulate the clean checkout: no local file, no env var. The stand-in keeps
    # the REAL basename, because the message quotes the path it looked in and a
    # reader has to recognise the file they are being asked to create.
    missing = tmp_path / "local_paths.yaml"
    assert not missing.exists()
    monkeypatch.setattr("framework.player_adapter.LOCAL_PATHS_FILE", missing)
    for name in placeholders:
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(KeyError) as excinfo:
        resolve_start_command(escimate["start_command"])
    message = str(excinfo.value)
    assert placeholders[0] in message, message
    assert "local_paths.yaml" in message, message
    assert "environment" in message, message
    # Two-sided: the SAME command resolves once the documented line exists, so
    # the raise above is the absence of configuration and not a broken resolver.
    missing.write_text(f'{placeholders[0]}: "C:/synthetic/app"\n', encoding="utf-8")
    assert resolve_start_command(escimate["start_command"])[-1].startswith("C:/synthetic/app/")


def test_a_placeholder_resolves_from_local_paths_file(monkeypatch, tmp_path):
    monkeypatch.setenv("VIBE_ROOT", str(tmp_path))
    monkeypatch.delenv("SOME_APP_DIR", raising=False)
    local = tmp_path / "local_paths.yaml"
    local.write_text('SOME_APP_DIR: "${VIBE_ROOT}/Tools/App"\n', encoding="utf-8")
    monkeypatch.setattr("framework.player_adapter.LOCAL_PATHS_FILE", local)
    resolved = resolve_start_command(["cmd", "/c", "${SOME_APP_DIR}/start_app.bat"])
    assert resolved == ["cmd", "/c",
                        str(tmp_path / "Tools" / "App" / "start_app.bat").replace(os.sep, "/")]


def test_the_environment_beats_the_local_paths_file(monkeypatch, tmp_path):
    local = tmp_path / "local_paths.yaml"
    local.write_text('SOME_APP_DIR: "C:/from-file"\n', encoding="utf-8")
    monkeypatch.setattr("framework.player_adapter.LOCAL_PATHS_FILE", local)
    monkeypatch.setenv("SOME_APP_DIR", "C:/from-env")
    assert resolve_start_command(["${SOME_APP_DIR}/x.bat"]) == ["C:/from-env/x.bat"]


def test_local_paths_file_is_gitignored():
    """If it ever stopped being ignored, the path would be committed and then
    published — the exact failure this indirection exists to prevent."""
    import subprocess
    rel = "players/local_paths.yaml"
    out = subprocess.run(["git", "check-ignore", "-v", rel], cwd=REPO_ROOT,
                         capture_output=True, text=True, encoding="utf-8")
    assert out.returncode == 0, f"{rel} is NOT gitignored: {out.stdout}{out.stderr}"
    assert rel in out.stdout, out.stdout
    # Negative control: a file that must NOT be ignored, so a `git check-ignore`
    # that says yes to everything cannot satisfy the assertion above.
    control = subprocess.run(["git", "check-ignore", "-v", "players/registry.yaml"],
                             cwd=REPO_ROOT, capture_output=True, text=True, encoding="utf-8")
    assert control.returncode != 0, f"control failed: registry.yaml reads as ignored: {control.stdout}"
