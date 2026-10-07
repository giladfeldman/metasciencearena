"""No source or doc file in the tree may contain a raw NUL byte.

Scope is the working TREE — tracked files and untracked-but-not-ignored ones
alike. See `_tracked_text_files` for why the untracked half is load-bearing: a
new file is exactly where a NUL gets introduced, and it is invisible to
`git ls-files` until the commit that ships it.

Why this is worth a test (2026-08-14, it bit twice in one session):

`leaderboard-app/scripts/lib/report.mjs` carried `const SEP = "<NUL>"` — a literal
0x00 used as a composite-key separator. One such byte makes the ENTIRE file read as
**binary** to grep, ripgrep and `git diff`. A content search for the cost field
returned "no matches found" while the field sat plainly in the file, which is part
of why a real defect in the engine that publishes reports stayed hidden.

Then, while writing the LESSONS.md entry about that, a Python `"\\u0000"` in the
generating script was interpreted as an actual NUL and written straight into the
lesson — the same trap, inside its own description.

The fix in every case is the six-character escape, which denotes the identical
value in JS, TS and Python while keeping the file text. Verified for the one place
where the NUL was a deliberate test input
(`lib/__tests__/signin-callback-url.test.ts`, a hostile-callback-URL security
test): `"\\u0000".charCodeAt(1) === 0`, and its 7 tests stayed green.

Binary assets are excluded by extension; this is about files humans and greps read.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

#: Extensions where a NUL is always a mistake — text humans and tools search.
TEXT_SUFFIXES = {
    ".md", ".py", ".mjs", ".js", ".ts", ".tsx", ".json", ".yaml", ".yml",
    ".txt", ".toml", ".cfg", ".sh", ".r", ".R", ".css", ".html",
    # `.jsonl` carries the run records and task sets — the format where a NUL
    # matters MOST, because it makes ripgrep treat the file as binary and skip
    # it, so a held-out leak sweep would silently pass over the very files that
    # hold the answers. `.ps1` is a real source format here too. Both were
    # missing from the original set. (Fable 5 cross-review, 2026-08-15.)
    ".jsonl", ".ps1",
}


def _tracked_text_files() -> list[Path]:
    """Files git tracks OR would track, filtered to text extensions.

    Uses git so generated and ignored output (node_modules, .next, .venv,
    build/) is excluded by definition rather than by a hand-maintained skip list
    that would rot.

    `--others --exclude-standard` is why this says "or would track". Until
    2026-09-07 the scan was `git ls-files -z` alone — TRACKED files only — so a
    brand-new source file was structurally invisible to it, and a NUL could be
    caught only one commit AFTER it was introduced. Measured that day: a new
    vitest file written in this repo carried three raw NULs (a composite key
    typed with the literal separator instead of the escape), `grep` reported
    "Binary file ... matches", and this test passed. Adding the untracked half
    makes the guard see a file at the moment it is written, which is the only
    moment a session is still there to fix it.

    Same shape as the article-custody lesson: a scan scoped to what git can see
    reports on GIT, not on the tree.
    """
    files = []
    for args in (["git", "ls-files", "-z"],
                 ["git", "ls-files", "-z", "--others", "--exclude-standard"]):
        out = subprocess.run(
            args, cwd=REPO, capture_output=True, text=True, check=True
        ).stdout
        for rel in out.split("\0"):
            if not rel:
                continue
            p = REPO / rel
            if p.suffix in TEXT_SUFFIXES and p.is_file():
                files.append(p)
    return files


def test_no_text_file_in_the_tree_contains_a_nul_byte():
    files = _tracked_text_files()
    # Guard against a vacuous pass: an empty file list would make this green while
    # checking nothing, which is the exact failure mode this repo keeps rediscovering.
    assert len(files) > 200, f"only {len(files)} text files found — the scan is not running"

    offenders = []
    for p in files:
        try:
            data = p.read_bytes()
        except OSError:  # pragma: no cover - unreadable file is a different problem
            continue
        n = data.count(b"\x00")
        if n:
            i = data.find(b"\x00")
            line = data[:i].count(b"\n") + 1
            offenders.append(f"{p.relative_to(REPO).as_posix()}:{line} ({n} NUL byte(s))")

    assert offenders == [], (
        "raw NUL byte(s) in text file(s) — this makes the whole file read as "
        "BINARY to grep/ripgrep/git diff, so content searches silently skip it:\n  "
        + "\n  ".join(offenders)
        + "\nUse the six-character escape instead; it denotes the same value in JS, TS "
          "and Python and keeps the file searchable."
    )


@pytest.mark.parametrize("lang", ["js", "py"])
def test_the_escape_denotes_the_same_value_as_a_raw_nul(lang: str):
    """The fix must be value-preserving, not merely cosmetic.

    Pinned because the NUL in signin-callback-url.test.ts is a deliberate hostile
    input: if the escape did not denote NUL, that security test would silently stop
    testing what it claims to.
    """
    escape = chr(92) + "u0000"
    if lang == "py":
        assert eval(f'"{escape}"') == "\x00"  # noqa: S307 - fixed literal
    else:
        # `which` FIRST. The old guard only inspected the return code, so an
        # environment with no node at all raised FileNotFoundError out of
        # subprocess.run instead of skipping — an error, not a skip, on any box
        # without node. Matches the `shutil.which("node") is None` idiom the
        # other node-dependent suites here already use.
        if shutil.which("node") is None:
            pytest.skip("node not available")
        node = subprocess.run(
            ["node", "-e", f'process.stdout.write(String("{escape}".charCodeAt(0)))'],
            capture_output=True, text=True,
        )
        if node.returncode != 0:  # node present but refused to run
            pytest.skip(f"node failed: {node.stderr.strip()[:200]}")
        assert node.stdout.strip() == "0"


def test_the_scan_reaches_UNTRACKED_files(tmp_path_factory):
    """The untracked half is what makes this guard able to see a NEW file.

    Pinned because it was proved once, with a planted NUL canary, and that
    canary was then deleted — leaving nothing to stop a future edit reverting to
    `git ls-files` alone. Measured 2026-09-07 at the real call site: tracked-only
    scanned 1,630 files and found nothing; the untracked half scanned 6 and
    caught the canary.

    Writes no NUL: a NUL file inside the repo would trip the very gate under
    test if this crashed. It asserts the FILE SET instead, which is the thing
    that regressed, and asserts both halves so a change that drops either one
    fails here.
    """
    probe = REPO / f".pytest-untracked-probe-{id(tmp_path_factory)}.py"
    probe.write_text("# transient probe; no NUL byte\n", encoding="utf-8")
    try:
        rel = {p.relative_to(REPO).as_posix() for p in _tracked_text_files()}
        assert probe.relative_to(REPO).as_posix() in rel, (
            "the scan no longer reaches untracked files — a NUL in a brand-new "
            "source file would be invisible until the commit that ships it"
        )
        # Two-sided: the tracked half must still be there too.
        assert "framework/report.py" in rel
    finally:
        probe.unlink(missing_ok=True)
