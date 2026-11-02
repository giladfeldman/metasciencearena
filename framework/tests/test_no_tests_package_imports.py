"""No test module may import from a package named `tests`.

Ten arenas ship `arenas/<id>/tests/__init__.py`, so in a full `pytest` run the name
`tests` can resolve to one of THEM. On 2026-10-30 five power-review modules did
`from tests.power_review_baselines import ...`: each passed alone, but the full run
collected 5149 tests with 5 errors and dropped all three power-review arenas' tests
(ModuleNotFoundError: No module named 'tests.power_review_baselines'). Load a shared
helper by file path instead.
"""
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_TESTS_IMPORT = re.compile(r"^\s*(from\s+tests(\.\w+)*\s+import\b|import\s+tests(\.\w+)*\b)", re.MULTILINE)


def _find_tests_package_imports(root: Path) -> list[str]:
    hits = []
    for path in sorted(root.rglob("test_*.py")):
        if any(part in {"node_modules", ".venv", "tmp", "_archive"} for part in path.parts):
            continue
        for match in _TESTS_IMPORT.finditer(path.read_text(encoding="utf-8", errors="replace")):
            hits.append(f"{path.relative_to(root).as_posix()}: {match.group(0).strip()}")
    return hits


def test_no_test_module_imports_from_a_tests_package():
    hits = _find_tests_package_imports(REPO / "tests") + _find_tests_package_imports(REPO / "arenas")
    assert hits == [], "load shared test helpers by file path, never via `tests.`:\n" + "\n".join(hits)
