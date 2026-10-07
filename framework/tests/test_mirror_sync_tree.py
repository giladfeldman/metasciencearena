"""The mirror sync must delete what we withdrew and NOTHING else.

WHY THIS TEST EXISTS
--------------------
This is the step that writes the public repository. It is irreversible in the
sense that matters — a wrong deletion is published and then has to be noticed.

Two real defects motivated it, both found on 2026-08-19:

1. The step it replaces used `rsync -a --delete`, and **rsync does not exist on
   this machine**. The sync therefore failed at the copy step on every run,
   AFTER the leak gate had passed and printed its success — which is why the
   public mirror sat at 56% drift (227 of 497 files missing) while the repo
   believed it had a working local sync path.
2. The replacement's first version deleted the public repo's `.gitignore`,
   because no manifest entry produces it. Without that file the public repo
   starts tracking `__pycache__/` and `.venv/`. Caught by reading a dry run's
   output, not by any automated check — hence this test.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_SUBJECT = REPO_ROOT / "scripts" / "mirror_sync_tree.py"

# `scripts/` is NOT part of the public manifest but `framework/tests/` is, so in
# the mirrored (public-package) run this file exists and its subject does not.
# Skipping is right here — the sync tool is an operator script for the machine
# that owns the corpora, and a public consumer neither has it nor needs it.
# Collected-but-unrunnable would abort the whole public suite, which is exactly
# the "a smaller green run" failure this repo has shipped before.
pytestmark = pytest.mark.skipif(
    not _SUBJECT.is_file(),
    reason="scripts/mirror_sync_tree.py is not part of the public package",
)

if _SUBJECT.is_file():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from mirror_sync_tree import sync  # noqa: E402


def _write(root: Path, rel: str, text: str = "x") -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def test_withdrawn_files_are_deleted(tmp_path):
    stage, dest = tmp_path / "stage", tmp_path / "dest"
    _write(stage, "framework/keep.py", "new")
    _write(dest, "framework/keep.py", "old")
    _write(dest, "framework/tests/test_audit.py", "withdrawn")

    sync(stage, dest)

    assert (dest / "framework/keep.py").read_text(encoding="utf-8") == "new"
    assert not (dest / "framework/tests/test_audit.py").exists(), (
        "a file removed from the manifest must disappear from the public repo, "
        "or a withdrawn file is published forever"
    )


def test_the_public_repos_own_gitignore_survives(tmp_path):
    """No manifest entry produces it, and deleting it makes the repo track junk."""
    stage, dest = tmp_path / "stage", tmp_path / "dest"
    _write(stage, "LICENSE", "MIT")
    _write(dest, ".gitignore", "__pycache__/\n.venv/\n")

    sync(stage, dest)

    assert (dest / ".gitignore").exists(), ".gitignore was deleted from the public repo"
    assert (dest / ".gitignore").read_text(encoding="utf-8") == "__pycache__/\n.venv/\n"


def test_git_is_never_touched_and_github_is_copy_only(tmp_path):
    stage, dest = tmp_path / "stage", tmp_path / "dest"
    _write(stage, ".github/workflows/ci.yml", "ours")
    _write(dest, ".git/HEAD", "ref: refs/heads/main")
    _write(dest, ".github/FUNDING.yml", "theirs")

    sync(stage, dest)

    assert (dest / ".git/HEAD").exists(), ".git must never be touched"
    assert (dest / ".github/FUNDING.yml").exists(), (
        ".github is copy-only — the public repo may keep things there we do not own"
    )
    assert (dest / ".github/workflows/ci.yml").read_text(encoding="utf-8") == "ours"


def test_unchanged_files_are_not_rewritten(tmp_path):
    """Only real changes should move mtimes, so git's dirty check stays meaningful."""
    stage, dest = tmp_path / "stage", tmp_path / "dest"
    _write(stage, "a.py", "same")
    _write(dest, "a.py", "same")
    _write(stage, "b.py", "new")

    written, deleted, unchanged = sync(stage, dest)
    assert (written, deleted, unchanged) == (1, 0, 1)


def test_directories_left_empty_are_pruned(tmp_path):
    stage, dest = tmp_path / "stage", tmp_path / "dest"
    _write(stage, "keep.py", "x")
    _write(dest, "keep.py", "x")
    _write(dest, "gone/deep/old.py", "x")

    sync(stage, dest)

    assert not (dest / "gone").exists(), "an emptied directory should not linger"
