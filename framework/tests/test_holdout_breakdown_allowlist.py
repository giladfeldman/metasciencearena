"""What may survive `score.breakdown` redaction on a held-out record.

Context. Until 2026-08-21 the whole breakdown was wiped to `{}` for held-out
records, keeping only `score.primary`. That was right when the breakdown carried
character counts and nothing else. Task set v2 of pdf-text-fidelity-v1 exists
precisely to publish **recall and precision separately** -- the finding was that
the v1 ranking was the precision ranking and the exact reverse of the recall
ranking -- and the held-out split is the half where document scope actually
varies. A blanket wipe would have hidden the finding on the data that produced it.

So the wipe became an ALLOWLIST of bounded ratios. The rule that decides
membership: does the value say anything about the SIZE or CONTENT of a document
we deliberately do not distribute? Counts and lengths do; ratios do not.

These tests pin the boundary in both directions, because an allowlist that lets
one count through is worse than the blanket wipe it replaced.
"""
from __future__ import annotations

from framework.holdout import (
    HELD_OUT,
    HELD_OUT_SAFE_BREAKDOWN_KEYS,
    REDACTED_INPUT_HASH,
    redact_held_out_record,
)


def _held_out_record(breakdown: dict) -> dict:
    return {
        "run_id": "r1",
        "arena_id": "pdf-text-fidelity-v1",
        "task_set_version": "v2",
        "task_id": "pmc-PMC0000001",
        "task_visibility": HELD_OUT,
        "player_id": "some-tool",
        "player_version": "1.0",
        "player_type": "tool",
        "input_hash": "deadbeef",
        "output": {"full_text": "the entire held-out paper"},
        "score": {"primary": 0.89, "breakdown": dict(breakdown)},
        "timestamp_utc": "2026-08-21T00:00:00Z",
    }


def test_bounded_ratios_survive_so_recall_and_precision_stay_publishable():
    rec = redact_held_out_record(_held_out_record({
        "body_recall": 0.986,
        "body_precision": 0.673,
        "leak_per_1k_body": 412.0,
        "gold_target_used": "ascii_greek",
    }))
    bd = rec["score"]["breakdown"]
    assert bd["body_recall"] == 0.986
    assert bd["body_precision"] == 0.673
    assert bd["leak_per_1k_body"] == 412.0
    assert bd["gold_target_used"] == "ascii_greek"
    # The rest of the boundary is untouched.
    assert rec["output"] == {}
    assert rec["input_hash"] == REDACTED_INPUT_HASH


def test_every_count_and_length_is_still_dropped():
    """THE regression. A token count is a length oracle for a corpus we do not ship.

    `holdout.py` already drops `usage` because `prompt_tokens` tracks the input
    document almost linearly. `n_tokens_must_include` is the same measurement
    under a different name, and `leak_by_category` is a dict of them.
    """
    rec = redact_held_out_record(_held_out_record({
        "body_recall": 0.986,
        "n_tokens_player": 13691,
        "n_tokens_must_include": 4808,
        "n_tokens_in_scope_hit": 4741,
        "n_tokens_leaked": 1982,
        "n_tokens_optional": 1161,
        "n_tokens_unattributed": 170,
        "n_chars_player": 84213,
        "n_chars_gold": 30011,
        "leak_by_category": {"reference": 1869},
        "optional_by_category": {"table_cell": 468},
    }))
    bd = rec["score"]["breakdown"]
    assert bd == {"body_recall": 0.986}, bd


def test_an_unknown_breakdown_key_is_redacted_by_default():
    """A scorer that adds a key gets it dropped until someone opts it in.

    Fail-safe direction: over-redacting is recoverable, publishing a held-out
    measurement is not.
    """
    rec = redact_held_out_record(_held_out_record({
        "body_recall": 0.9,
        "some_future_metric": 1234,
        "sample_missing_text": "a verbatim sentence from the held-out paper",
    }))
    assert rec["score"]["breakdown"] == {"body_recall": 0.9}


def test_the_allowlist_contains_no_count_or_length_key():
    """Pins the rule itself, not just today's contents.

    Anyone adding `n_tokens_*` or `*_by_category` to the allowlist has to delete
    this test to do it, which is the point.
    """
    for key in HELD_OUT_SAFE_BREAKDOWN_KEYS:
        assert not key.startswith("n_"), key
        assert not key.endswith("_by_category"), key
        assert "count" not in key, key
        assert "chars" not in key, key
        assert "tokens" not in key, key


def test_an_error_marker_still_replaces_the_whole_breakdown():
    """An errored record keeps only the sanitized error TYPE, as before."""
    rec = redact_held_out_record(_held_out_record({
        "error": "TimeoutError: reading /held/out/PMC0000001.pdf failed",
        "body_recall": 0.5,
    }))
    bd = rec["score"]["breakdown"]
    assert set(bd) == {"error"}
    assert "PMC0000001" not in bd["error"]


def test_a_public_record_is_untouched():
    rec = dict(_held_out_record({"body_recall": 0.9, "n_tokens_player": 100}))
    rec["task_visibility"] = "public"
    out = redact_held_out_record(rec)
    assert out["score"]["breakdown"] == {"body_recall": 0.9, "n_tokens_player": 100}
    assert out["output"] == {"full_text": "the entire held-out paper"}
