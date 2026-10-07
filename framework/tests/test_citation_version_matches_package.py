"""`CITATION.cff` must declare the version the package actually is.

WHY THIS FILE EXISTS. `CITATION.cff` is how a researcher cites this software,
and it is MIRRORED to the public repository — so a stale version there is a
wrong published claim, not an internal untidiness. It drifted silently:
measured 2026-09-02, `pyproject.toml` said 0.4.0 while `CITATION.cff` said
0.3.0, released "2026-08-07". Nothing compared them, so nothing went red.

This is the class the 2026-09-02 project review named as the dominant one: an
invariant that everybody agrees with, written down nowhere executable. A
version constant copied into a second file is
`.claude/skills/_project/lessons.md`'s "one value, two facts" — the copy is a
label that can silently disagree with the thing it labels.

The check is deliberately two-sided. A regex that stops matching would make
both sides `None` and the equality trivially true, which is the false-green
shape this repo has hit 41 times; `test_the_probe_is_not_vacuous` asserts each
side actually parsed a plausible version before the agreement is believed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"
CITATION = REPO_ROOT / "CITATION.cff"

_SEMVER = re.compile(r"^\d+\.\d+\.\d+")


def _pyproject_version() -> str | None:
    for line in PYPROJECT.read_text(encoding="utf-8").splitlines():
        m = re.match(r'^version\s*=\s*"([^"]+)"', line.strip())
        if m:
            return m.group(1)
    return None


def _citation_version() -> str | None:
    for line in CITATION.read_text(encoding="utf-8").splitlines():
        m = re.match(r'^version:\s*"?([^"\s]+)"?', line.strip())
        if m:
            return m.group(1)
    return None


@pytest.mark.skipif(
    not (PYPROJECT.exists() and CITATION.exists()),
    reason="one of the two version files is absent from this checkout",
)
def test_the_probe_is_not_vacuous() -> None:
    """Both sides must actually parse, or the agreement below means nothing."""
    py, cff = _pyproject_version(), _citation_version()
    assert py is not None, f"no version parsed from {PYPROJECT}"
    assert cff is not None, f"no version parsed from {CITATION}"
    assert _SEMVER.match(py), f"pyproject version is not semver-shaped: {py!r}"
    assert _SEMVER.match(cff), f"CITATION version is not semver-shaped: {cff!r}"


@pytest.mark.skipif(
    not (PYPROJECT.exists() and CITATION.exists()),
    reason="one of the two version files is absent from this checkout",
)
def test_citation_version_matches_pyproject() -> None:
    """The cited version IS the package version. Both files are published."""
    py, cff = _pyproject_version(), _citation_version()
    assert cff == py, (
        f"CITATION.cff declares version {cff!r} but pyproject.toml is {py!r}. "
        "CITATION.cff is mirrored publicly, so this publishes a wrong citation. "
        "Bump CITATION.cff (and its date-released) in the same commit as any "
        "version change."
    )
