"""Smoke test for the pcurve R tool player (p-curve-v1).

Pipes every revealed task envelope through players/adapters/pcurve.R and asserts:
  - returncode 0,
  - stdout parses as JSON,
  - the JSON validates against the arena's output schema (Draft 2020-12),
  - the R p-curve verdict matches the arena's computed gold, and its right_skew_p
    agrees with the Python gold within tolerance (the cross-validation guarantee: the
    deterministic R reference reproduces the canonical computed gold).

Skipped when Rscript is unavailable. The adapter is a pure base-R implementation of
the Simonsohn et al. (2014) p-curve, so it needs no extra R packages. Invokes plain
`Rscript` (no --vanilla) to mirror the runner's RCliAdapter exactly.
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
ADAPTER = REPO / "players" / "adapters" / "pcurve.R"
ARENA_DIR = REPO / "arenas" / "p-curve-v1"

from players.adapters.tests._rscript_probe import RSCRIPT_UNAVAILABLE_REASON, rscript_path

# Resolve the SAME way the runner does (explicit > RSCRIPT_BINARY > PATH), not
# via a bare PATH lookup — see conftest.py for why that mattered.
RSCRIPT = rscript_path()
pytestmark = pytest.mark.skipif(RSCRIPT is None, reason=RSCRIPT_UNAVAILABLE_REASON)


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "_pcurve_gen_smoke", ARENA_DIR / "generator.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_pcurve_gen_smoke"] = mod
    spec.loader.exec_module(mod)
    return mod


def _validator():
    from jsonschema import Draft202012Validator
    schema = json.loads(
        (ARENA_DIR / "schemas" / "output.schema.json").read_text(encoding="utf-8")
    )
    return Draft202012Validator(schema)


def test_pcurve_adapter_smoke():
    generator = _load_generator()
    validator = _validator()

    envelopes = list(generator.generate("v1", seed=0, split="revealed"))
    assert envelopes, "no revealed tasks generated"

    for env in envelopes:
        task_id = env["task_id"]
        gold = generator.ground_truth(task_id)
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
        # Cross-validation: the R p-curve reproduces the computed gold verdict ...
        assert out["evidential_value"] == gold["evidential_value"], (
            f"{task_id} ({gold['label']}): R verdict={out['evidential_value']} "
            f"gold={gold['evidential_value']} (R right_skew_p={out.get('right_skew_p')})"
        )
        # ... and its right_skew_p matches the Python gold to high precision.
        assert abs(out["right_skew_p"] - gold["right_skew_p"]) <= 1e-4, (
            f"{task_id}: R right_skew_p={out['right_skew_p']} "
            f"gold={gold['right_skew_p']}"
        )
        assert out["confidence"] == 1.0
