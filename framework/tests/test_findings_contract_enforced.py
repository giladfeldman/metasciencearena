"""The findings contract must be LOADED, not merely shipped.

`contract/schemas/findings.schema.json` has existed since the contract was
written. It declares `additionalProperties: false` and an exact field list. But
`run_record.schema.json` declares `score.findings.items` as a bare
`{"type": "object"}` and names the real schema only inside a `description`
string — so nothing ever loaded it, and it was documentation wearing a schema's
filename.

Two things drifted through that gap and neither failed anything:

  1. `code-translation-r-v1` emitted an undeclared `detail` field. The task page
     then rendered `detail` and NOTHING ELSE, so for the other 21 arenas a reader
     saw a severity chip and a generic sentence while `anchor`/`evidence`/
     `correct_value` were dropped — 97% of findings.
  2. `effect-size-conversion-v1` emitted a numeric `evidence` against a
     string-typed declaration.

Both were reconciled 2026-08-12 (the schema was wrong about `evidence`; `detail`
is legitimate and is now declared), all 2599 existing findings arrays were
verified to validate, and only then was enforcement switched on.

Regression target: `framework/runner.py::_validate_findings`, called from
`_play_one` immediately after `scorer.score()`.
"""
from __future__ import annotations

import pytest

from framework import runner


def test_valid_findings_pass_including_the_reconciled_shapes():
    """Everything real scorers emit today must still be accepted."""
    runner._validate_findings(
        {
            "findings": [
                # the common shape: where, what gold said, what the player said
                {"category": "grim_missed", "anchor": {"stat_id": "s1"},
                 "evidence": "3.47", "correct_value": "3.48"},
                # effect-size-conversion-v1 emits a bare NUMBER as evidence
                {"category": "value_wrong", "evidence": 0.4416, "correct_value": 0.44},
                # code-translation-r-v1 emits a pre-written sentence
                {"category": "exec_mismatch",
                 "detail": "'n_missing_after' = 0, gold = 21."},
                # aggregated occurrences
                {"category": "greek_dropped", "count": 12, "examples": ["η²p", "χ²"]},
                # held-out findings survive redaction as category+count only
                {"category": "grim_missed", "count": 3},
            ]
        },
        "test-arena",
        "t1",
    )


def test_undeclared_field_is_rejected():
    """The exact drift that let `detail` in unnoticed for months."""
    with pytest.raises(ValueError, match="findings_schema_violation"):
        runner._validate_findings(
            {"findings": [{"category": "x", "undeclared_field": 1}]},
            "test-arena", "t1",
        )


def test_missing_required_category_is_rejected():
    """`category` joins the finding to arena.yaml#error_categories; without it the
    UI cannot resolve a severity or a description, so the finding is unreadable."""
    with pytest.raises(ValueError, match="findings_schema_violation"):
        runner._validate_findings({"findings": [{"evidence": "oops"}]}, "test-arena", "t1")


def test_wrong_type_is_rejected():
    with pytest.raises(ValueError, match="findings_schema_violation"):
        runner._validate_findings(
            {"findings": [{"category": "x", "count": "not-an-integer"}]},
            "test-arena", "t1",
        )


def test_absent_findings_is_not_an_error():
    """`findings` is optional — most scorers emit none on a clean task."""
    runner._validate_findings({"primary": 1.0}, "test-arena", "t1")
    runner._validate_findings({}, "test-arena", "t1")


def test_the_schema_is_actually_loaded_from_package_data():
    """Guards the failure mode itself: a schema that exists but is never read.

    If someone re-points this at an inline dict, or the packaged schema goes
    missing, this fails rather than silently accepting everything again.
    """
    validator = runner._findings_validator()
    schema = validator.schema
    assert schema.get("items", {}).get("additionalProperties") is False, (
        "findings.schema.json must keep additionalProperties:false — it is the "
        "only thing that catches an undeclared field"
    )
    assert "detail" in schema["items"]["properties"]
    assert set(schema["items"]["required"]) == {"category"}


# --- task envelope contract (2026-08-12) --------------------------------------
#
# Same gap, other direction: the runner validated the player's OUTPUT but never
# the TASK it handed over. `task_envelope.schema.json` declares five required
# fields and additionalProperties:false, and nothing loaded it either.
#
# Verified against all 22 arenas' generated envelopes before enforcing. Note the
# PDF arenas need PYTHONPATH=<repo root> to import at all — an earlier check
# without it reported a clean pass on 17 of 22 and called that "safe to enforce",
# which is the false-green-from-empty-input trap in CLAUDE.md.

VALID_ENVELOPE = {
    "task_id": "t1",
    "arena_id": "test-arena",
    "task_set_version": "v1",
    "difficulty": {"tier": 1},
    "visibility": "public",
    "split": "revealed",
    "input": {"text": "hello"},
}


def test_valid_envelope_passes():
    runner._validate_envelope(dict(VALID_ENVELOPE), "test-arena")


def test_envelope_missing_required_field_is_rejected():
    for field in ("task_id", "arena_id", "task_set_version", "difficulty", "input"):
        env = {k: v for k, v in VALID_ENVELOPE.items() if k != field}
        with pytest.raises(ValueError, match="task_envelope_schema_violation"):
            runner._validate_envelope(env, "test-arena")


def test_envelope_with_undeclared_key_is_rejected():
    """A misspelled or stray key must not ride along unnoticed."""
    env = dict(VALID_ENVELOPE, visibilty="public")  # note the typo
    with pytest.raises(ValueError, match="task_envelope_schema_violation"):
        runner._validate_envelope(env, "test-arena")


def test_every_real_arena_generator_emits_a_valid_envelope():
    """The check that makes the two above meaningful rather than hypothetical.

    Skips arenas whose generator cannot import (optional heavy deps); asserts we
    still covered a substantial number, so this can never quietly degrade into
    validating nothing at all.
    """
    import importlib.util
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    checked = 0
    for arena_dir in sorted((repo / "arenas").iterdir()):
        gen = arena_dir / "generator.py"
        if not gen.exists():
            continue
        spec = importlib.util.spec_from_file_location(f"_env_gen_{arena_dir.name}", gen)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
            envelopes = list(module.generate("v1", 0))
        except Exception:
            continue  # heavy/optional dependency absent in this environment
        for env in envelopes:
            runner._validate_envelope(env, arena_dir.name)
        checked += 1
    assert checked >= 15, (
        f"only {checked} arena generators were exercised — this test is supposed "
        "to cover the real corpus, not silently shrink to nothing"
    )
