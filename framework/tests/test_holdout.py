"""Unit tests for the framework.holdout redaction contract."""
from __future__ import annotations

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
