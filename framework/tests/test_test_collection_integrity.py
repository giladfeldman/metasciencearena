"""The suite must not silently shrink.

WHAT HAPPENED (2026-08-07)
--------------------------
`pytest -q` reported a confident **1023 passed** for months. It was collecting
**1023 of 1143** tests. 19 arenas each ship `tests/test_generator.py` and
`tests/test_scorer.py`; only 11 of those `tests/` dirs had an `__init__.py`, and
the arena dirs above them are hyphenated (`stats-extraction-v1`), so no valid
package path could be formed either way. Every file resolved to the same module
name, the first import won, and every later arena's file was collected as a
duplicate of it — 120 tests never ran.

That is the dangerous shape: not a red suite, a **smaller green one**. One of the
hidden tests was genuinely failing (a "clean control" rendering "the test was
significant" beside p = 0.057), and another was broken against a corpus added
after it was written.

Underneath that sat a second collision of the same kind: several arena tests did
`sys.path.insert(0, ARENA_DIR)` then `import scorer`, registering
`sys.modules["scorer"]`. All 19 arenas have a `scorer.py`, so once collection was
fixed those tests began exercising **the wrong arena's scorer**.

Both are pinned below. Neither is exotic — they are the ordinary consequence of
many sibling directories sharing filenames, which this repo will keep doing as
arenas are added.
"""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
ARENAS = REPO / "arenas"

pytestmark = pytest.mark.skipif(not ARENAS.is_dir(), reason="needs the arenas tree")


def test_pytest_is_configured_to_give_every_test_module_a_unique_name():
    """`consider_namespace_packages` is what recovers the 120 dropped tests.

    Without it, collection is 1025; with it, 1144. `--import-mode=importlib`
    alone is NOT enough — that was already set the whole time this was broken.
    """
    cfg = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    ini = cfg["tool"]["pytest"]["ini_options"]
    assert "--import-mode=importlib" in ini.get("addopts", ""), (
        "addopts lost --import-mode=importlib"
    )
    assert ini.get("consider_namespace_packages") is True, (
        "consider_namespace_packages is off. With it off, arena test files that "
        "share a filename collapse onto one module name and 120 tests silently "
        "stop running while the suite still reports green."
    )


def test_no_arena_test_imports_its_arena_module_under_a_bare_name():
    """`import scorer` after a path insert makes tests exercise another arena.

    The module name `scorer` is shared by all 19 arenas. Whichever arena is
    imported first wins `sys.modules["scorer"]`, and every later arena's tests
    then assert against code that is not theirs — silently, and while passing.

    The fix every arena should use is a unique name:

        scorer = _load("_my_arena_scorer", "scorer.py")
    """
    bare = re.compile(r"^\s*import\s+(scorer|generator|_normalize|label_map)\b", re.M)
    offenders: list[str] = []
    checked = 0
    for path in sorted(ARENAS.glob("*/tests/test_*.py")):
        src = path.read_text(encoding="utf-8")
        checked += 1
        if "sys.path.insert" not in src:
            continue
        for m in bare.finditer(src):
            offenders.append(f"{path.relative_to(REPO).as_posix()}: import {m.group(1)}")

    # Non-vacuity: a glob that matched nothing would pass this trivially.
    assert checked > 10, f"only {checked} arena test files found — the glob is wrong"
    assert offenders == [], (
        "these tests import an arena module under a name every arena shares, so "
        "they can silently test the WRONG arena's code:\n  "
        + "\n  ".join(offenders)
        + "\nLoad it under a unique name instead, e.g. "
          '`scorer = _load(\"_my_arena_scorer\", \"scorer.py\")`.'
    )


def test_every_arena_with_a_scorer_has_tests_for_it():
    """A cheap coverage floor, so a new arena cannot arrive untested.

    Not a quality bar — just "somebody wrote a test file". It exists because the
    collision above meant an arena could LOOK tested (its file was collected)
    while its tests never ran.
    """
    missing = [
        d.name
        for d in sorted(ARENAS.iterdir())
        if (d / "scorer.py").is_file() and not list(d.glob("tests/test_*.py"))
    ]
    assert missing == [], f"arenas with a scorer but no tests at all: {missing}"
