"""Smoke test for the effectsize-convert R tool player (effect-size-conversion-v1).

Pipes every revealed task envelope through players/adapters/effectsize_convert.R and
asserts:
  - returncode 0,
  - stdout parses as JSON,
  - the JSON validates against the arena's output schema (Draft 2020-12),
  - the converted value matches the arena's computed gold within tolerance (the
    cross-validation guarantee: the deterministic tool reproduces the canonical gold).

Skipped when Rscript or the effectsize package is unavailable. Invokes plain `Rscript`
(no --vanilla) to mirror the runner's RCliAdapter exactly.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
ADAPTER = REPO / "players" / "adapters" / "effectsize_convert.R"
ARENA_DIR = REPO / "arenas" / "effect-size-conversion-v1"

from players.adapters.tests._rscript_probe import RSCRIPT_UNAVAILABLE_REASON, rscript_path

# Resolve the SAME way the runner does (explicit > RSCRIPT_BINARY > PATH), not
# via a bare PATH lookup — see conftest.py for why that mattered.
RSCRIPT = rscript_path()
pytestmark = pytest.mark.skipif(RSCRIPT is None, reason=RSCRIPT_UNAVAILABLE_REASON)


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "_esconv_gen_smoke", ARENA_DIR / "generator.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_esconv_gen_smoke"] = mod
    spec.loader.exec_module(mod)
    return mod


def _r_package_installed(pkg: str) -> bool:
    try:
        proc = subprocess.run(
            [RSCRIPT, "-e",
             f'quit(status = if (requireNamespace("{pkg}", quietly=TRUE)) 0 else 1)'],
            capture_output=True, text=True, timeout=120,
        )
        return proc.returncode == 0
    except Exception:
        return False


def _validator():
    from jsonschema import Draft202012Validator
    schema = json.loads(
        (ARENA_DIR / "schemas" / "output.schema.json").read_text(encoding="utf-8")
    )
    return Draft202012Validator(schema)


def test_effectsize_convert_adapter_smoke():
    if not _r_package_installed("effectsize"):
        pytest.skip("R package effectsize not installed")
    generator = _load_generator()
    validator = _validator()

    envelopes = list(generator.generate("v1", seed=0, split="revealed"))
    assert envelopes, "no revealed tasks generated"

    for env in envelopes:
        task_id = env["task_id"]
        gold = generator.ground_truth(task_id)["converted"]
        proc = subprocess.run(
            [RSCRIPT, str(ADAPTER)],
            input=json.dumps(env),
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=300,
        )
        assert proc.returncode == 0, (
            f"{task_id} exited {proc.returncode}: {proc.stderr[-500:]}"
        )
        out = json.loads(proc.stdout)
        errors = sorted(validator.iter_errors(out), key=lambda e: list(e.path))
        assert not errors, (
            f"{task_id} output failed schema validation:\n"
            + "\n".join(f"  - {e.message} (at {list(e.path)})" for e in errors)
            + f"\nraw: {json.dumps(out)}"
        )
        # Cross-validation: the tool reproduces the computed gold.
        tol = max(0.01, 0.01 * abs(gold))
        assert abs(out["converted"] - gold) <= tol, (
            f"{task_id} ({env['input']['from']}->{env['input']['to']}): "
            f"tool={out['converted']} gold={gold} (tol={tol})"
        )
        assert out["confidence"] == 1.0
