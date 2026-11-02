"""The reproduce commands the leaderboard site prints are accepted by `framework run`.

The task page's "Reproduce this run" card printed
`python -m framework run --arena A --player P --task T`, which the parser rejects
("ambiguous option: --task could match --task-set, --tasks-file"), and the arena's
"How scoring works" page hardcoded `--task-set v1` for every arena. Both commands
live in TypeScript; this test reads them from the source and parses them with the
real CLI parser, so a CLI change that breaks the site's instructions goes red here.
"""
from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest

from framework.cli import build_parser

APP = Path(__file__).resolve().parents[2] / "leaderboard-app"
CARD = APP / "components" / "task-detail" / "reproduce-card.tsx"
OPEN_BENCHMARK = APP / "app" / "arenas" / "[arena_id]" / "open-benchmark" / "page.tsx"
#: The site is not part of the public mirror (mirrored-suite run, 2026-11-02): there the two tests
#: that read its pages skip by name instead of failing on a missing file.
NEEDS_SITE = pytest.mark.skipif(not APP.is_dir(), reason="leaderboard-app/ is not in the public mirror -- NOT verified here")

SAMPLE = {
    "a.arenaId": "stats-extraction-v1", "arena_id": "stats-extraction-v1",
    "a.taskSetVersion": "v5", "taskSetVersion": "v5",
    "a.playerId": "statcheck", "a.split": "revealed",
}


def _fill(template: str) -> list[str]:
    """Substitute `${expr}` placeholders with sample values and split like a shell."""
    def sub(m: re.Match[str]) -> str:
        expr = m.group(1).strip()
        if expr not in SAMPLE:
            raise AssertionError(f"unexpected placeholder ${{{expr}}} in {template!r}")
        return SAMPLE[expr]
    return shlex.split(re.sub(r"\$\{([^}]+)\}", sub, template))


def _card_template() -> str:
    src = CARD.read_text(encoding="utf-8")
    body = re.search(r"export function reproduceCommand\([^)]*\)[^{]*\{(.*?)\n\}", src, re.S)
    assert body, "reproduceCommand() not found in reproduce-card.tsx"
    parts = re.findall(r"`([^`]*)`", body.group(1))
    assert parts, "reproduceCommand() builds no template literal"
    return "".join(parts)


def _open_benchmark_run_line() -> str:
    src = OPEN_BENCHMARK.read_text(encoding="utf-8")
    lines = [l for l in re.findall(r"`([^`]*)`", src) if l.startswith("python -m framework run")]
    assert len(lines) == 1, f"expected one `framework run` line, found {lines}"
    return lines[0]


def _parse(argv: list[str]):
    assert argv[:3] == ["python", "-m", "framework"], argv
    return build_parser().parse_args(argv[3:])


@NEEDS_SITE
def test_task_page_reproduce_command_parses():
    args = _parse(_fill(_card_template()))
    assert args.func.__name__ == "_cmd_run"
    assert (args.arena, args.task_set, args.players, args.split) == (
        "stats-extraction-v1", "v5", ["statcheck"], "revealed")
    assert args.tasks_file == Path("task.txt")


@NEEDS_SITE
def test_open_benchmark_run_command_parses_and_uses_the_current_task_set():
    line = _open_benchmark_run_line()
    assert "--task-set v1" not in line, "task set must not be hardcoded"
    args = _parse(_fill(line.replace("<player_id>", "statcheck")))
    assert args.func.__name__ == "_cmd_run" and args.task_set == "v5"


def test_the_old_card_command_is_rejected():
    """Control: the parse check can fail. The pre-fix command must not parse."""
    old = "python -m framework run --arena stats-extraction-v1 --player statcheck --task t-x"
    with pytest.raises(SystemExit):
        _parse(shlex.split(old))
