"""A schema-rejected answer is kept for diagnosis, and never crosses held-out (d-a25d35).

Until schema v11 the runner stored `output: {}` on an output_schema_violation, so
the value that broke the schema was lost; the desktop seat had a testable
hypothesis (rationale emitted as null) and no record to test it on
(conductor F-0082). Driven through `run_tournament`, not a hand-built record.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from jsonschema import Draft202012Validator

from framework.holdout import held_out_leak_reasons, redact_held_out_record

REPO = Path(__file__).resolve().parents[2]
SCHEMA = json.loads(
    (REPO / "framework" / "contract" / "schemas" / "run_record.schema.json")
    .read_text(encoding="utf-8"))


def _run(tmp_path, arena_dir, registry_path, split) -> list[dict]:
    from framework.runner import run_tournament
    from framework.storage import read_records

    out = tmp_path / f"runs-{split}.jsonl"
    run_tournament(
        arena_dir=arena_dir, task_set_version="v1",
        registry_path=registry_path,
        player_ids=["stub-pass", "stub-schema-violation"],
        output_path=out, trials=1, split=split,
    )
    return read_records(out)


def _strict_split_arena(tmp_path, fake_split_arena_dir) -> Path:
    """fake_split_arena with fake_arena's closed output schema.

    The split fixture is the only one with a public split, and its schema allows
    extra keys, so the violation stub would conform there. A tmp copy keeps the
    shared fixture untouched.
    """
    arena = tmp_path / "fake_split_arena"
    shutil.copytree(fake_split_arena_dir, arena,
                    ignore=shutil.ignore_patterns("__pycache__"))
    schema_path = arena / "schemas" / "output.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema["additionalProperties"] = False
    schema_path.write_text(json.dumps(schema), encoding="utf-8")
    return arena


def test_rejected_answer_is_kept_on_public_tasks_only(
        tmp_path, fake_split_arena_dir, fake_registry_path):
    arena = _strict_split_arena(tmp_path, fake_split_arena_dir)
    public = _run(tmp_path, arena, fake_registry_path, "revealed")
    held = _run(tmp_path, arena, fake_registry_path, "private")
    viol_pub = [r for r in public if r["player_id"] == "stub-schema-violation"]
    viol_held = [r for r in held if r["player_id"] == "stub-schema-violation"]
    # Controls: both splits really produced a schema violation.
    assert viol_pub and viol_held
    for r in viol_pub + viol_held:
        assert r["task_visibility"] in ("public", "held_out")
        assert r["score"]["breakdown"]["error"].startswith("output_schema_violation")

    for r in viol_pub:
        assert r["task_visibility"] == "public"
        assert r["output"] == {}
        assert isinstance(r.get("rejected_output"), dict) and r["rejected_output"]
    for r in viol_held:
        assert r["task_visibility"] == "held_out"
        assert "rejected_output" not in r
    # Accepted answers carry no rejected_output: the field means "refused".
    assert all("rejected_output" not in r
               for r in public + held if r["player_id"] == "stub-pass")

    validator = Draft202012Validator(SCHEMA)
    for r in public + held:
        assert not list(validator.iter_errors(r))


def test_redaction_drops_it_and_the_leak_check_sees_it():
    rec = {"task_visibility": "held_out", "output": {}, "input_hash": "x",
           "rejected_output": {"rationale": None, "text": "verbatim document text"},
           "score": {"primary": 0.0, "breakdown": {"error": "output_schema_violation: x"}}}
    assert any("rejected_output" in r for r in held_out_leak_reasons(rec, kind="record"))
    redacted = redact_held_out_record(rec)
    assert "rejected_output" not in redacted
    assert not any("rejected_output" in r for r in held_out_leak_reasons(redacted, kind="record"))
