"""Every path that EXECUTES model output runs with an allowlisted environment (d-a07ebd).

df65c7b found code-translation-r-v1 running the player's R with `{**os.environ}`:
the private seed and the real profile reached model-written code. The audit that
followed (2026-10-08) found that r_runner.py is the only arena module that starts
a process at all; every model CLI adapter spawns through `hermetic.spawn_kwargs`.

This pins both facts, so a new arena that executes model output cannot do it
silently: it must be added below AND use `hermetic.untrusted_code_env`.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

_SPAWN = re.compile(r"\bsubprocess\.(run|Popen|call|check_call|check_output)\b|\bos\.system\(")

#: Arena modules allowed to start a process, each because it runs MODEL-WRITTEN code
#: under `hermetic.untrusted_code_env`.
EXEC_MODULES = {"arenas/code-translation-r-v1/r_runner.py"}


def _arena_modules():
    for p in sorted((REPO / "arenas").glob("*/*.py")):
        if p.name.startswith("test_"):
            continue
        yield p.relative_to(REPO).as_posix(), p.read_text(encoding="utf-8")


def test_only_listed_arena_modules_start_processes():
    found = {rel for rel, src in _arena_modules() if _SPAWN.search(src)}
    assert found, "control: the scan found no process call at all -- the pattern is broken"
    assert found == EXEC_MODULES, (
        f"arena modules starting processes changed: {sorted(found ^ EXEC_MODULES)}. "
        "If one runs model output, give it hermetic.untrusted_code_env and list it here.")


def test_listed_exec_modules_use_the_untrusted_env_and_not_ours():
    for rel in EXEC_MODULES:
        src = (REPO / rel).read_text(encoding="utf-8")
        assert "untrusted_code_env" in src, rel
        code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
        assert "**os.environ" not in code, rel
