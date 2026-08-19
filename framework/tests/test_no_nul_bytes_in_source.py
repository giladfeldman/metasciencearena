"""No tracked source or doc file may contain a raw NUL byte.

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
    """Files git tracks, filtered to text extensions. Uses git so generated and
    ignored output (node_modules, .next, .venv, build/) is excluded by definition
    rather than by a hand-maintained skip list that would rot."""
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout
    files = []
    for rel in out.split("\0"):
        if not rel:
            continue
        p = REPO / rel
        if p.suffix in TEXT_SUFFIXES and p.is_file():
            files.append(p)
    return files


def test_no_tracked_text_file_contains_a_nul_byte():
    files = _tracked_text_files()
    # Guard against a vacuous pass: an empty file list would make this green while
    # checking nothing, which is the exact failure mode this repo keeps rediscovering.
    assert len(files) > 200, f"only {len(files)} tracked text files found — the scan is not running"

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
        "raw NUL byte(s) in tracked text file(s) — this makes the whole file read as "
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
        node = subprocess.run(
            ["node", "-e", f'process.stdout.write(String("{escape}".charCodeAt(0)))'],
            capture_output=True, text=True,
        )
        if node.returncode != 0:  # node unavailable in this environment
            pytest.skip("node not available")
        assert node.stdout.strip() == "0"
