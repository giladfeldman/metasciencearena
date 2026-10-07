"""The stamped `record["adherence"]` block, and the two ways it could be wrong.

WHY IT IS STAMPED AT ALL. Everything in the block is derivable from
``score.breakdown.error`` for a PUBLIC record, and `framework/adherence.py`
derives it — which is what gives the metric retrospective coverage over every
record ever written. The block exists for the one case derivation cannot reach:
``holdout.sanitize_error`` reduces a held-out record's error to its exception
TYPE, so a later reader recovers the verdict ("the schema rejected this") and
never the diagnosis ("it put a `verdict` value in the `kind` field"). Since the
Innsbruck marathon runs both splits, that would have lost exactly the evidence
the metric was built to collect.

THE TWO FAILURES THIS FILE PINS.

1. **Drift.** Stamped and derived are two implementations of one classification.
   If they diverge, a record disagrees with the report built from it and nothing
   fails. `test_the_stamped_block_agrees_with_the_derived_one` runs both over the
   same record and compares.

2. **An unbounded channel out of a held-out task.** `classes` is a CLOSED enum in
   the schema on purpose. The 2026-08-07 cross-model review found
   ``score.breakdown.error`` acting as a free-text escape from held-out tasks —
   an assertion message could carry gold verbatim and pass every other check.
   A new field on the same record is the same hazard with a new name.
   `test_no_stamped_class_is_outside_the_declared_vocabulary` is the guard.

Watched failing before `runner._play_one` stamped anything.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from framework import adherence
from framework.holdout import redact_held_out_record

REPO = Path(__file__).resolve().parents[2]
SCHEMA = json.loads(
    (REPO / "framework" / "contract" / "schemas" / "run_record.schema.json")
    .read_text(encoding="utf-8"))

STATS_SCHEMA = json.loads(
    (REPO / "arenas" / "stats-extraction-v1" / "schemas" / "output.schema.json")
    .read_text(encoding="utf-8"))
STATS_VOCAB = adherence.enum_values_in_schema(STATS_SCHEMA)

#: Real, verbatim messages from this repo's committed run records.
REAL = [
    "output_schema_violation: 'not_a_stat' is not one of ['nhst_stat', 'effect_size', 'confidence_interval']",
    "output_schema_violation: [0.22, 0.54] is not of type 'object'",
    "output_schema_violation: 'confidence' is a required property",
    "output_schema_violation: Additional properties are not allowed ('confidence_note' was unexpected)",
    "output_schema_violation",  # the held-out, sanitized form
    "ReadTimeout: HTTPSConnectionPool(host='integrate.api.nvidia.com', port=443)",
    None,  # a clean, scored record
]


def _record(error, *, visibility="public"):
    breakdown = {"error": error} if error else {}
    rec = {
        "run_id": "r", "arena_id": "stats-extraction-v1", "task_set_version": "v5",
        "task_id": "t", "player_id": "p", "player_version": "1", "player_type": "ai-model",
        "input_hash": "h", "output": {}, "timestamp_utc": "2026-09-05T00:00:00Z",
        "task_visibility": visibility,
        "score": {"primary": 0.0 if error else 0.5, "breakdown": breakdown},
    }
    rec["adherence"] = {
        "outcome": adherence.outcome(rec),
        "classes": adherence.classify_record(rec, enum_values=STATS_VOCAB),
    }
    return rec


def test_the_schema_declares_the_block_and_closes_both_of_its_enums():
    prop = SCHEMA["properties"]["adherence"]
    assert prop["additionalProperties"] is False
    assert set(prop["properties"]["outcome"]["enum"]) == {
        adherence.CONFORMANT, adherence.VIOLATION, adherence.UNATTRIBUTED}
    # A closed vocabulary, matching the module's, so a new class cannot reach a
    # published record without the schema saying so.
    assert set(prop["properties"]["classes"]["items"]["enum"]) == set(adherence.CLASS_PRECEDENCE)


def test_the_block_is_not_required_so_old_records_stay_valid():
    """5,386 records predate `schema_version` entirely. Requiring the block would
    assert retroactively that most of the corpus was always malformed — the same
    reasoning that kept `instrument` out of required[] in v7."""
    assert "adherence" not in SCHEMA["required"]


def test_the_stamped_block_agrees_with_the_derived_one():
    for err in REAL:
        rec = _record(err)
        assert rec["adherence"]["outcome"] == adherence.outcome(rec), err
        assert rec["adherence"]["classes"] == adherence.classify_record(
            rec, enum_values=STATS_VOCAB), err


def test_no_stamped_class_is_outside_the_declared_vocabulary():
    declared = set(SCHEMA["properties"]["adherence"]["properties"]["classes"]["items"]["enum"])
    seen = set()
    for err in REAL:
        seen |= set(_record(err)["adherence"]["classes"])
    assert seen, "control: the fixtures must produce at least one class"
    assert seen <= declared, f"unbounded class(es) reached a record: {sorted(seen - declared)}"


def test_a_held_out_record_keeps_the_diagnosis_the_message_body_loses():
    """The whole reason the block is stamped rather than only derived.

    Two-sided: the redacted record must LOSE the message (control that redaction
    really happened) and KEEP the class (the thing this field buys).
    """
    rec = _record(
        "output_schema_violation: 'not_a_stat' is not one of ['nhst_stat', "
        "'effect_size', 'confidence_interval']",
        visibility="held_out")
    assert rec["adherence"]["classes"] == ["enum_swap"]

    redacted = redact_held_out_record(rec)
    # Control: the free-text diagnosis really is gone.
    assert redacted["score"]["breakdown"] == {"error": "output_schema_violation"}
    assert adherence.classify_record(redacted, enum_values=STATS_VOCAB) == ["redacted"]
    # And the stamped class survived, because it is a bounded categorical.
    assert redacted["adherence"]["classes"] == ["enum_swap"]
    assert redacted["adherence"]["outcome"] == adherence.VIOLATION


def test_the_runner_stamps_it_on_every_record_it_writes():
    """A field nothing writes is a declaration, not a capability.

    Greps the real call site rather than a copy: `record["adherence"] = ` must
    appear in `_play_one`, unconditionally, before the record is returned.
    """
    src = (REPO / "framework" / "runner.py").read_text(encoding="utf-8")
    assert 'record["adherence"] = {' in src
    # The version that introduced `adherence` was 8, and every later version
    # still carries it. Pinning the EXACT number here made an unrelated schema
    # addition red (v10, player_identity) while saying nothing about adherence --
    # a test failing for a reason it was not written to catch. Assert the floor
    # this file is actually about, and let the schema keep moving.
    m = re.search(r"^RECORD_SCHEMA_VERSION = (\d+)$", src, re.MULTILINE)
    assert m, "runner.py no longer declares RECORD_SCHEMA_VERSION"
    assert int(m.group(1)) >= 8


# ---------------------------------------------------------------- real call site
# Everything above builds a record by hand. That measures the helper, not the
# runner. These two drive `run_tournament` end to end, through the same
# `_play_one` a real tournament uses, so the assertion is about what actually
# gets written to disk.
#
# The 2026-08-22 lesson behind this: a standalone probe reported 9 of 225 players
# checked; the CLI reported 127. The 9 was manufactured by the harness.

def test_the_runner_really_writes_the_block(tmp_path, fake_arena_dir, fake_registry_path):
    from framework.runner import run_tournament
    from framework.storage import read_records

    out = tmp_path / "runs.jsonl"
    run_tournament(
        arena_dir=fake_arena_dir,
        task_set_version="v1",
        registry_path=fake_registry_path,
        player_ids=["stub-pass", "stub-fail", "stub-schema-violation"],
        output_path=out,
        trials=1,
    )
    records = read_records(out)
    assert records, "control: the tournament wrote nothing"
    by_player: dict[str, list[dict]] = {}
    for r in records:
        by_player.setdefault(r["player_id"], []).append(r)

    # Every record carries the block, and its schema_version says so.
    #
    # `>= 8`, not `== 8`: the adherence block arrived in v8 and every version after
    # it still carries one. Pinning equality made this test assert "nothing has been
    # added to the record since", which is a claim about the whole schema that this
    # file has no business making — it went red on the unrelated v9
    # `provenance.request_max_tokens` addition (2026-09-11). The shape of the record
    # is pinned deliberately, once, by
    # test_run_record_schema.py::test_schema_shape_change_requires_a_version_bump.
    for r in records:
        assert r["schema_version"] >= 8
        assert set(r["adherence"]) == {"outcome", "classes"}

    # The three outcomes are really distinguished — a classifier that returned
    # one constant would pass a weaker version of this test.
    assert {r["adherence"]["outcome"] for r in by_player["stub-pass"]} == {"conformant"}
    assert {r["adherence"]["outcome"] for r in by_player["stub-fail"]} == {"unattributed"}
    viol = by_player["stub-schema-violation"]
    assert {r["adherence"]["outcome"] for r in viol} == {"violation"}
    assert all(r["adherence"]["classes"] == ["extra_property"] for r in viol)
    # And a conformant record carries no class, rather than an empty-string one.
    assert all(r["adherence"]["classes"] == [] for r in by_player["stub-pass"])


def test_a_written_record_still_validates_against_the_contract(
        tmp_path, fake_arena_dir, fake_registry_path):
    """The new field must not make the runner's own output schema-invalid.

    `run_record.schema.json` has `additionalProperties: false`, so an unlisted
    top-level key would reject every record the runner writes — and nothing else
    in the suite validates a FRESHLY WRITTEN record against the contract.
    """
    from jsonschema import Draft202012Validator

    from framework.runner import run_tournament
    from framework.storage import read_records

    out = tmp_path / "runs.jsonl"
    run_tournament(
        arena_dir=fake_arena_dir, task_set_version="v1",
        registry_path=fake_registry_path,
        player_ids=["stub-pass", "stub-schema-violation"],
        output_path=out, trials=1,
    )
    records = read_records(out)
    assert records
    validator = Draft202012Validator(SCHEMA)
    for r in records:
        errors = sorted(validator.iter_errors(r), key=lambda e: e.path)
        assert not errors, [e.message for e in errors]
