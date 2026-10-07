"""No file the public mirror ships may name the person who builds it.

WHY THIS FILE EXISTS. `arenas/open-practices-repro-v1/catalogs/repo_templates.yaml`
carried the owner's real Windows profile path as a "synthetic" bad-path example,
and it reached the public mirror. `scan_for_leaks` could not see it: it compares
mirrored files against private seeds and held-out text, and a personal path is
neither. Found by an outside scan of the public repo on 2026-09-27; the same
check then found the owner's first name in a POSIX home path in
`framework/tests/test_host.py`.

NO REAL NAME IS WRITTEN HERE. This file is itself mirrored, so a test asserting
on the literal would publish it. The identity is read from the machine at run
time (`publish.personal_identity`), and the unit tests use an invented account.

Two-sided throughout: every "clean" assertion is paired with a control proving
the same code flags the thing when it IS there -- a scan that reads nothing, or
a pattern that matches nothing, reports the same green as a clean tree.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from framework import publish

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "contract" / "public_manifest.txt"
NAMES, EMAILS = {"zqxowner"}, {"zqx.owner@example.org"}


def _hits(text: str) -> list[str]:
    return publish.personal_path_hits(text, NAMES, EMAILS)


@pytest.mark.parametrize("text", [
    'abs_dirs: ["C:/Users/zqxowner/Dropbox/thesis/raw"]',
    r"see C:\Users\zqxowner\Documents\x.csv",
    r"C:\\Users\\zqxowner\\x",                  # JSON-escaped backslashes
    "/home/zqxowner/models/x.gguf",
    "/c/Users/zqxowner/Vibe",                   # Git Bash form
    "C:/users/ZQXOWNER/x",                      # case-insensitive
    "mailto: zqx.owner@example.org",
])
def test_a_personal_path_or_email_is_flagged(text):
    assert _hits(text), text


@pytest.mark.parametrize("text", [
    "C:/Users/student/Dropbox/thesis/raw",      # the replacement placeholder
    "C:/Users/zqxownerX/x",                     # a longer, different account
    "profiling zqxowner as a bare word",        # not path-shaped
    "/home/labuser/analysis",
    "",
])
def test_an_unrelated_path_or_bare_word_is_not_flagged(text):
    assert not _hits(text), text


def test_generic_account_names_are_dropped(monkeypatch):
    """A CI runner's home is /home/runner; flagging it would only cry wolf."""
    monkeypatch.setattr(publish.Path, "home", classmethod(lambda cls: Path("/home/runner")))
    monkeypatch.setattr("getpass.getuser", lambda: "runner")
    names, _ = publish.personal_identity(REPO)
    assert "runner" not in names


def _no_home(cls):
    raise RuntimeError("Could not determine home directory.")


def test_an_unset_home_falls_back_to_getuser(monkeypatch):
    """No HOME/USERPROFILE must not crash the gate when the account is still readable."""
    monkeypatch.setattr(publish.Path, "home", classmethod(_no_home))
    monkeypatch.setattr("getpass.getuser", lambda: "zqxowner")
    names, _ = publish.personal_identity(REPO)
    assert "zqxowner" in names


def test_no_readable_account_refuses_to_certify(monkeypatch):
    """With no account name at all the path check would match nothing -- a false
    green -- so the gate must refuse loudly instead."""
    monkeypatch.setattr(publish.Path, "home", classmethod(_no_home))

    def _no_user():
        raise OSError("no user")

    monkeypatch.setattr("getpass.getuser", _no_user)
    with pytest.raises(RuntimeError, match="refusing to certify"):
        publish.personal_identity(REPO)


def _mirrored_files() -> list[Path]:
    """The published file set: the manifest in the private repo; in the public
    checkout, which does not ship the manifest, every file git tracks there."""
    if MANIFEST.is_file():
        # Strip CR: the manifest has been CRLF before, and `[ -f "$f\r" ]` once
        # opened zero files and reported the mirror clean.
        lines = MANIFEST.read_text(encoding="utf-8").splitlines()
        return [Path(l.strip()) for l in lines if l.strip() and not l.startswith("#")]
    out = subprocess.run(["git", "ls-files"], cwd=str(REPO),
                         capture_output=True, text=True, encoding="utf-8").stdout or ""
    return [Path(l) for l in out.splitlines() if l.strip()]


def test_the_mirrored_file_list_is_not_vacuous():
    files = _mirrored_files()
    assert len(files) > 200
    assert sum((REPO / f).is_file() for f in files) == len(files)


def test_no_mirrored_file_names_this_machines_user():
    names, emails = publish.personal_identity(REPO)
    if not names and not emails:
        pytest.skip("no identifying account name or git identity on this machine")
    bad = {}
    for rel in _mirrored_files():
        p = REPO / rel
        if p.suffix not in publish._TEXT_SUFFIXES:
            continue
        found = publish.personal_path_hits(p.read_text(encoding="utf-8", errors="replace"),
                                           names, emails)
        if found:
            bad[rel.as_posix()] = len(found)
    assert not bad, f"mirrored files naming this machine's user: {bad}"


def test_the_live_identity_is_caught_when_it_is_there(tmp_path):
    """Positive control for the test above, with the REAL identity: build the
    offending shape at run time and prove the gate flags it."""
    names, emails = publish.personal_identity(REPO)
    if not names:
        pytest.skip("no identifying account name on this machine")
    name = sorted(names)[0]
    assert publish.personal_path_hits(f'"C:/Users/{name}/Dropbox/thesis/raw"', names, emails)


def test_scan_for_leaks_refuses_a_file_with_a_personal_path(tmp_path, monkeypatch):
    """The gate the mirror build actually runs, not just the helper."""
    monkeypatch.setattr(publish, "personal_identity", lambda root: (NAMES, EMAILS))
    (tmp_path / "bad.yaml").write_text('p: "C:/Users/zqxowner/x"\n', encoding="utf-8")
    (tmp_path / "good.yaml").write_text('p: "C:/Users/student/x"\n', encoding="utf-8")
    bad = publish.scan_for_leaks(tmp_path, [Path("bad.yaml")], gates="allowlist")
    good = publish.scan_for_leaks(tmp_path, [Path("good.yaml")], gates="allowlist")
    assert len(bad) == 1 and "bad.yaml" in bad[0]
    assert good == []
