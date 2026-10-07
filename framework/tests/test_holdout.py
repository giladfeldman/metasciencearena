"""Unit tests for the framework.holdout redaction contract."""
from __future__ import annotations

import json

from framework.holdout import (
    gold_keys,
    held_out_leak_reasons,
    is_held_out,
    redact_ground_truth_entry,
    redact_held_out_record,
    strip_gold,
)


def test_is_held_out_defaults_to_true_when_unlabeled() -> None:
    assert is_held_out({"visibility": "held_out"}) is True
    assert is_held_out({"visibility": "public"}) is False
    assert is_held_out({}) is True            # safe direction: over-redact
    assert is_held_out(None) is True


def test_gold_keys_and_strip_gold() -> None:
    gt = {"doc_id": "x", "section_count": 3, "gold_sections": [1], "gold_full_text": "S"}
    assert sorted(gold_keys(gt)) == ["gold_full_text", "gold_sections"]
    assert strip_gold(gt) == {"doc_id": "x", "section_count": 3}
    assert strip_gold(None) == {}


def test_redact_ground_truth_entry_strips_held_out_gold_keeps_metadata() -> None:
    entry = {
        "envelope": {"visibility": "held_out", "difficulty": {"a": 1}},
        "ground_truth": {"doc_id": "PMC1", "source_kind": "real-pmc",
                         "section_count": 3, "gold_sections": [1, 2], "gold_full_text": "SECRET"},
    }
    out = redact_ground_truth_entry(entry)
    assert out["ground_truth"] == {"doc_id": "PMC1", "source_kind": "real-pmc", "section_count": 3}
    assert out["envelope"]["difficulty"] == {"a": 1}        # metadata preserved
    assert held_out_leak_reasons(out, kind="ground_truth") == []
    # original object not mutated
    assert "gold_full_text" in entry["ground_truth"]


def test_redact_ground_truth_entry_passes_public_through() -> None:
    entry = {"envelope": {"visibility": "public"}, "ground_truth": {"gold_x": "KEEP"}}
    assert redact_ground_truth_entry(entry)["ground_truth"] == {"gold_x": "KEEP"}


def test_redact_held_out_record_blanks_output_hash_breakdown_keeps_primary() -> None:
    rec = {
        "task_visibility": "held_out",
        "output": {"full_text": "SECRET", "sections": [1]},
        "input_hash": "deadbeef",
        "score": {"primary": 0.8, "breakdown": {"heading_f1": 0.5, "n_sections_gold": 4},
                  "findings": [{"category": "c", "count": 2}]},
    }
    out = redact_held_out_record(rec)
    assert out["output"] == {}
    assert out["input_hash"] == "<redacted>"                # schema needs the key
    assert out["score"]["breakdown"] == {}
    assert out["score"]["primary"] == 0.8                   # ranking survives
    assert out["score"]["findings"] == [{"category": "c", "count": 2}]  # untouched
    assert held_out_leak_reasons(out, kind="record") == []
    # original not mutated
    assert rec["output"]["full_text"] == "SECRET"


def test_redact_held_out_record_keeps_error_TYPE_but_drops_the_message() -> None:
    """The error marker survives redaction; its message body does not.

    This test previously asserted the FULL message was preserved
    ("TimeoutError: boom"). That was the defect, not the contract: the ``error``
    key exempts a held-out breakdown from every other check in the module, so
    an unbounded string there is an uncontrolled channel out of a held-out task.
    Found by cross-model review (codex, 2026-08-07) and reproduced — see
    ``test_held_out_error_message_body_is_a_leak`` below for the payload that
    used to pass clean.

    The type is kept because that is the part maintainers actually act on
    (timeout vs HTTP vs scorer crash), and it carries no task content.
    """
    rec = {"task_visibility": "held_out", "output": {},
           "score": {"primary": 0.0, "breakdown": {"error": "TimeoutError: boom"}}}
    out = redact_held_out_record(rec)
    assert out["score"]["breakdown"] == {"error": "TimeoutError"}
    assert held_out_leak_reasons(out, kind="record") == []


def test_held_out_error_message_body_is_a_leak() -> None:
    """A gold value smuggled into an error string must NOT read as clean."""
    leaking = {"task_visibility": "held_out", "output": {}, "input_hash": "<redacted>",
               "score": {"primary": 0.0, "breakdown": {"error": "AssertionError: expected 'THE GOLD'"}}}
    reasons = held_out_leak_reasons(leaking, kind="record")
    assert reasons, "an error message body must be reported as a leak"
    assert "message body" in reasons[0]
    # And redaction must actually remove it.
    assert redact_held_out_record(leaking)["score"]["breakdown"] == {"error": "AssertionError"}


def test_held_out_error_breakdown_may_not_smuggle_other_keys() -> None:
    """`error` exempts the breakdown — it must not become a container."""
    rec = {"task_visibility": "held_out", "output": {}, "input_hash": "<redacted>",
           "score": {"primary": 0.0, "breakdown": {"error": "TimeoutError", "per_task": "SECRET"}}}
    reasons = held_out_leak_reasons(rec, kind="record")
    assert any("beside 'error'" in r for r in reasons), reasons


def test_redact_held_out_record_passes_public_through() -> None:
    rec = {"task_visibility": "public", "output": {"a": 1}, "input_hash": "h",
           "score": {"primary": 1.0, "breakdown": {"x": 1}}}
    out = redact_held_out_record(rec)
    assert out["output"] == {"a": 1}
    assert out["input_hash"] == "h"
    assert out["score"]["breakdown"] == {"x": 1}


def test_leak_reasons_flags_each_channel() -> None:
    leaky_gt = {"envelope": {"visibility": "held_out"}, "ground_truth": {"gold_x": 1}}
    assert held_out_leak_reasons(leaky_gt, kind="ground_truth")
    leaky_rec = {"task_visibility": "held_out", "output": {"a": 1}, "input_hash": "h",
                 "score": {"breakdown": {"k": 1}}}
    reasons = held_out_leak_reasons(leaky_rec, kind="record")
    assert any("output" in r for r in reasons)
    assert any("input_hash" in r for r in reasons)
    assert any("breakdown" in r for r in reasons)


def test_held_out_envelope_input_is_redacted():
    """The held-out INPUT can itself be the secret, not just the gold.

    Added 2026-08-04 with code-translation-r-v1's real `_held_out/` corpus. For a
    document arena the bytes were already omitted upstream, so stripping only
    `ground_truth` sufficed. But here the held-out SOURCE SCRIPT is the very
    thing players must not have seen — publishing it in the tracked dump would
    defeat the split entirely while every existing contamination test still
    passed (they only look at gold fields).
    """
    from framework.holdout import redact_ground_truth_entry, REDACTED_INPUT

    entry = {
        "envelope": {
            "task_id": "t-holdout", "visibility": "held_out",
            "difficulty": {"tier": 2},
            "input": {"source_language": "spss",
                      "source_code": "T-TEST PAIRS=pre WITH post (PAIRED).",
                      "required_statistics": ["t_statistic"]},
        },
        "ground_truth": {"tier": 2, "gold_statistics": {"t_statistic": -7.93}},
    }
    out = redact_ground_truth_entry(entry)

    assert out["envelope"]["input"]["source_code"] == REDACTED_INPUT
    assert not [k for k in out["ground_truth"] if k.startswith("gold")]
    # Non-revealing metadata must survive — the leaderboard renders the split.
    assert out["envelope"]["input"]["required_statistics"] == ["t_statistic"]
    assert out["envelope"]["difficulty"] == {"tier": 2}
    # And the original must not be mutated.
    assert entry["envelope"]["input"]["source_code"].startswith("T-TEST")


def test_public_envelope_input_is_untouched():
    """Redaction must not blank a PUBLIC task's input — players need it."""
    from framework.holdout import redact_ground_truth_entry

    entry = {
        "envelope": {"task_id": "t-pub", "visibility": "public",
                     "input": {"source_code": "DESCRIPTIVES VARIABLES=age."}},
        "ground_truth": {"gold_statistics": {"mean_age": 42.0}},
    }
    out = redact_ground_truth_entry(entry)
    assert out["envelope"]["input"]["source_code"] == "DESCRIPTIVES VARIABLES=age."
    assert out["ground_truth"]["gold_statistics"] == {"mean_age": 42.0}


# --- provenance.seed is a held-out leak channel (2026-08-09) ------------------


def test_redaction_strips_provenance_seed_for_held_out_records():
    """The private SEED is the entire secret, and it rode in `provenance.seed`.

    `redact_held_out_record` closed output / input_hash / score.breakdown but left
    `provenance.seed` untouched, and private run files ARE tracked (13 of them).
    That is half of the original stats-extraction-v1 exposure — the module docstring
    of `test_private_seed_not_in_tracked_files.py` names both channels: "36 tracked
    run records carry `provenance.seed` verbatim, and the same integer is embedded
    in every one of their `task_id`s".

    Task set v2 fixed the task_id channel (hashed discriminator) and, on the first
    private run, republished the freshly-rotated seed through this one. Anyone with
    repo access could read it straight out of the committed JSONL and regenerate the
    entire held-out split, gold included.

    Nothing legitimately consumes the raw seed from a record: reproducing a private
    run requires `.private_seed` regardless.
    """
    record = {
        "task_id": "t-tier1-d1-0-sabc12345",
        "task_visibility": "held_out",
        "output": {"x": 1},
        "input_hash": "a" * 64,
        "score": {"primary": 0.5, "breakdown": {"precision": 1.0}},
        "provenance": {"seed": 987654321, "split": "private", "host": "box"},
    }
    out = redact_held_out_record(record)
    assert out["provenance"]["seed"] != 987654321
    assert "987654321" not in json.dumps(out), "the seed survived somewhere in the record"
    # The key must remain so the record shape is stable for consumers.
    assert "seed" in out["provenance"]
    # Non-secret provenance is untouched.
    assert out["provenance"]["host"] == "box"
    assert out["provenance"]["split"] == "private"


def test_redaction_leaves_public_record_provenance_alone():
    """The revealed seed is committed in arena.yaml — publishing it aids reproduction."""
    record = {
        "task_id": "t-tier1-d1-0-s5feceb66",
        "task_visibility": "public",
        "output": {"x": 1},
        "input_hash": "b" * 64,
        "score": {"primary": 0.5, "breakdown": {"precision": 1.0}},
        "provenance": {"seed": 0, "split": "revealed"},
    }
    out = redact_held_out_record(record)
    assert out["provenance"]["seed"] == 0


# ---------------------------------------------------------------------------
# Regression: `usage` must be stripped from held-out records.
#
# Found 2026-08-14 while wiring token/cost capture through to the published
# mirror (R2). The run_record schema has documented since 2026-08-13 that a
# held-out record's `usage` is stripped "where prompt_tokens is a near-exact
# proxy for input document length" — but NEITHER redaction boundary implemented
# it (`framework.holdout.redact_held_out_record`, nor `redactHeldOutRecord` in
# leaderboard-app/scripts/build-data.mjs), and `leak_reasons` did not look for
# it either.
#
# Latent until now only because no record carried `usage` at all. Publishing
# tokens turns it into a live egress of a document-length proxy for every
# held-out PDF — i.e. the schema promised a redaction the code never performed.
# ---------------------------------------------------------------------------


def _held_out_record_with_usage() -> dict:
    return {
        "run_id": "r1",
        "arena_id": "pdf-text-fidelity-v1",
        "task_set_version": "v1",
        "task_id": "t1",
        "player_id": "p1",
        "player_version": "1.0.0",
        "player_type": "ai-model",
        "task_visibility": "held_out",
        "input_hash": "sha256:deadbeef",
        "output": {"text": "..."},
        "score": {"primary": 0.5, "breakdown": {"f1": 0.5}},
        "timestamp_utc": "2026-08-14T00:00:00Z",
        # prompt_tokens is a near-exact proxy for the length of the held-out PDF.
        "usage": {"prompt_tokens": 41234, "completion_tokens": 812, "total_tokens": 42046},
    }


def test_redact_held_out_record_strips_usage():
    out = redact_held_out_record(_held_out_record_with_usage())
    assert "usage" not in out, (
        "held-out record kept `usage`; prompt_tokens leaks input document length"
    )


def test_leak_reasons_flags_usage_on_held_out_record():
    reasons = held_out_leak_reasons(_held_out_record_with_usage(), kind="record")
    assert any("usage" in r for r in reasons), (
        f"held_out_leak_reasons ignored a held-out record carrying usage; got {reasons}"
    )


def test_redact_held_out_record_keeps_usage_on_public_record():
    rec = _held_out_record_with_usage()
    rec["task_visibility"] = "public"
    out = redact_held_out_record(rec)
    assert out.get("usage", {}).get("prompt_tokens") == 41234, (
        "public records must keep usage — that is the whole point of capturing it"
    )
