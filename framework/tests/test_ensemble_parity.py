"""The Python and JavaScript ensemble implementations must agree.

There are two on purpose: `framework/ensemble.py` is the reference and the local
CLI, while `leaderboard-app/scripts/lib/ensemble.mjs` runs inside the Node build
that Vercel actually executes. Publishing a number computed by a Python script
the deploy never runs is how the per-player feedback reports 404'd in production.

Two implementations invite drift, and drift here would mean the site publishes a
different ensemble number than the one anyone reproduces locally. So this pins
them against each other on the REAL corpus rather than a toy fixture — a parity
test over synthetic data would pass while the arenas disagreed.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from framework import ensemble

REPO = Path(__file__).resolve().parents[2]
JS_MODULE = REPO / "leaderboard-app" / "scripts" / "lib" / "ensemble.mjs"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None or not JS_MODULE.exists(),
    reason="node and leaderboard-app/scripts/lib/ensemble.mjs are both required",
)

TOL = 1e-9


def _js_analyse(arena_dir: Path) -> dict:
    """Run the JS implementation over the same tracked run records."""
    # The module specifier must be a file:// URL — Windows absolute paths are not
    # a supported ESM scheme ("Received protocol 'c:'").
    module_url = JS_MODULE.resolve().as_uri()
    script = f"""
import {{ readdirSync, statSync, readFileSync }} from "node:fs";
import {{ join }} from "node:path";
import {{ analyseEnsemble }} from {json.dumps(module_url)};

const arenaDir = {json.dumps(arena_dir.as_posix())};
const records = [];
function walk(dir) {{
  for (const name of readdirSync(dir)) {{
    const p = join(dir, name);
    if (statSync(p).isDirectory()) {{
      if (name === "_archive" || name === "_pilot_archive") continue;
      walk(p);
    }} else if (name.endsWith(".jsonl")) {{
      for (const line of readFileSync(p, "utf-8").split(/\\r?\\n/)) {{
        if (!line.trim()) continue;
        try {{ records.push(JSON.parse(line)); }} catch {{}}
      }}
    }}
  }}
}}
walk(join(arenaDir, "runs"));
process.stdout.write(JSON.stringify(analyseEnsemble({json.dumps(arena_dir.name)}, records)));
"""
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        capture_output=True, text=True, timeout=300, cwd=REPO,
    )
    if proc.returncode != 0:
        pytest.fail(f"node failed for {arena_dir.name}: {proc.stderr[:500]}")
    return json.loads(proc.stdout)


def _arenas_with_records():
    for d in sorted((REPO / "arenas").iterdir()):
        if (d / "arena.yaml").exists() and (d / "runs").exists():
            yield d


def test_python_and_js_agree_on_every_arena():
    compared = 0
    for arena_dir in _arenas_with_records():
        py = ensemble.analyse(arena_dir)
        js = _js_analyse(arena_dir)

        assert py["n_players"] == js["n_players"], f"{arena_dir.name}: player count"
        assert py["n_common_tasks"] == js["n_common_tasks"], f"{arena_dir.name}: task count"
        if "note" in py or "note" in js:
            assert ("note" in py) == ("note" in js), f"{arena_dir.name}: one side found nothing to ensemble"
            continue

        assert abs(py["oracle_all"] - js["oracle_all"]) < TOL, (
            f"{arena_dir.name}: oracle {py['oracle_all']} (py) vs {js['oracle_all']} (js)"
        )
        assert abs(py["headroom"] - js["headroom"]) < TOL, f"{arena_dir.name}: headroom"
        assert py["saturates_at_k"] == js["saturates_at_k"], (
            f"{arena_dir.name}: saturation {py['saturates_at_k']} (py) vs "
            f"{js['saturates_at_k']} (js)"
        )
        assert py["best_single"]["player_id"] == js["best_single"]["player_id"], (
            f"{arena_dir.name}: best single player disagrees"
        )
        # The greedy ORDER is the published narrative ("which tool to add next"),
        # so it must match element for element, not merely end at the same total.
        assert [r["player_id"] for r in py["greedy_curve"]] == \
               [r["player_id"] for r in js["greedy_curve"]], (
            f"{arena_dir.name}: greedy order disagrees"
        )
        compared += 1

    assert compared >= 10, (
        f"only {compared} arenas were actually compared — this test is supposed to "
        "cover the real corpus, not silently shrink to nothing"
    )
