"""The mirror builder must publish committed bytes only (d-d411b2).

On 2026-10-07 another session's uncommitted edit to
`players/adapters/_docling_common.py` reached the public sync's change list,
because the builder copies from the shared working tree. The builder now
refuses any manifest path that is untracked or differs from HEAD.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_SUBJECT = REPO_ROOT / "scripts" / "build_public_mirror.py"

# `scripts/` is not in the public manifest; see test_mirror_sync_tree.py.
pytestmark = pytest.mark.skipif(
    not _SUBJECT.is_file(),
    reason="scripts/build_public_mirror.py is not part of the public package",
)

if _SUBJECT.is_file():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from build_public_mirror import uncommitted_manifest_paths  # noqa: E402


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@example.org")
    _git(tmp_path, "config", "user.name", "t")
    for name in ("clean.py", "edited.py", "staged.py"):
        (tmp_path / name).write_text("v1\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def test_flags_untracked_edited_and_staged_but_not_clean(tmp_path):
    repo = _repo(tmp_path)
    (repo / "edited.py").write_text("v2\n", encoding="utf-8")
    (repo / "staged.py").write_text("v2\n", encoding="utf-8")
    _git(repo, "add", "staged.py")
    (repo / "new.py").write_text("v1\n", encoding="utf-8")

    bad = uncommitted_manifest_paths(
        repo, [Path(p) for p in ("clean.py", "edited.py", "staged.py", "new.py")])

    assert sorted(bad) == [
        "edited.py (uncommitted changes)",
        "new.py (untracked)",
        "staged.py (uncommitted changes)",
    ]


def test_clean_manifest_is_not_flagged(tmp_path):
    # The two-sided control: a zero here must be a real zero.
    repo = _repo(tmp_path)
    assert uncommitted_manifest_paths(repo, [Path("clean.py"), Path("edited.py")]) == []


def test_git_failure_raises_instead_of_reading_as_clean(tmp_path):
    with pytest.raises(RuntimeError):
        uncommitted_manifest_paths(tmp_path / "not-a-repo", [Path("x.py")])
