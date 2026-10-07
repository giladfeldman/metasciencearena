"""A truncation produced under a LOWER ceiling must not survive a retry at a higher one.

`retry-failed --include-truncated` exists to re-run tasks whose answer was cut off
after the ceiling has been raised. Until 2026-09-11 it then THREW THE RESULT AWAY
whenever the fresh attempt also failed:

    elif tid not in merged:      # tid IS already in merged, so this never fired
        merged[tid] = rec

so the published record stayed the attempt made under the ceiling now known to be
wrong, and carried no trace of having been retried. Only a console line said it
happened. Found independently by both seats of the 2026-09-11 consult round while
reviewing the Cohere Catalyst grant run, whose 5 truncations were all in exactly
that state: published at max_tokens=4000 against a measured provider maximum of
4096.

The fix is deliberately narrow. The surrounding rule -- "a fresh error never
downgrades an existing OK" -- is correct and load-bearing, and the replacement
fires ONLY when the kept record is truncated AND the fresh attempt used a
STRICTLY HIGHER ceiling. An equal ceiling does not qualify: that is a re-roll of a
non-deterministic player, which buys a different sample rather than better
evidence. (Measured the same week: re-rolling one such player moved a published
mean 0.5000 -> 0.2750.)
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _cli():
    spec = importlib.util.spec_from_file_location("sa_cli", REPO / "framework" / "cli.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _truncated(task_id: str, ceiling: int | None, primary: float = 0.0) -> dict:
    rec = {
        "task_id": task_id,
        "player_id": "p",
        "score": {"primary": primary, "breakdown": {}},
        "response_meta": {"finish_reason": "length"},
        "usage": {"completion_tokens": ceiling or 0},
    }
    if ceiling is not None:
        rec["provenance"] = {"request_max_tokens": ceiling}
    return rec


def _clean(task_id: str, ceiling: int | None, primary: float = 1.0) -> dict:
    rec = {
        "task_id": task_id,
        "player_id": "p",
        "score": {"primary": primary, "breakdown": {}},
        "response_meta": {"finish_reason": "stop"},
        "usage": {"completion_tokens": 40},
    }
    if ceiling is not None:
        rec["provenance"] = {"request_max_tokens": ceiling}
    return rec


def _merge(existing: dict, fresh: dict, *, inc_tr: bool) -> dict:
    """Reproduce the merge exactly as `_cmd_retry_failed` performs it."""
    cli = _cli()
    merged = dict(existing)
    for tid, rec in fresh.items():
        if cli._record_is_ok(rec, include_truncated=inc_tr):
            merged[tid] = rec
        elif tid not in merged:
            merged[tid] = rec
        elif (
            inc_tr
            and cli._record_is_truncated(merged[tid])
            and cli._record_ceiling(rec) is not None
            and cli._record_ceiling(rec) > (cli._record_ceiling(merged[tid]) or 0)
        ):
            merged[tid] = rec
    return merged


def test_ceiling_is_read_from_provenance_then_response_meta():
    cli = _cli()
    assert cli._record_ceiling({"provenance": {"request_max_tokens": 4096}}) == 4096
    assert cli._record_ceiling({"response_meta": {"max_tokens": 6000}}) == 6000
    # provenance wins when both are present
    assert cli._record_ceiling(
        {"provenance": {"request_max_tokens": 4096}, "response_meta": {"max_tokens": 4000}}
    ) == 4096
    # NEGATIVE CONTROL: absent must be None, never 0 -- a hosted player's record
    # carries neither field, and reading that as a zero ceiling would make every
    # fresh attempt look like an improvement.
    assert cli._record_ceiling({}) is None
    assert cli._record_ceiling({"provenance": {}, "response_meta": {}}) is None
    # A bool is not a ceiling.
    assert cli._record_ceiling({"provenance": {"request_max_tokens": True}}) is None


def test_a_failed_retry_at_a_HIGHER_ceiling_replaces_the_stale_truncation():
    """The defect. The fresh attempt is the one made under the correct config."""
    existing = {"t1": _truncated("t1", 4000)}
    fresh = {"t1": _truncated("t1", 4096)}
    merged = _merge(existing, fresh, inc_tr=True)
    assert merged["t1"]["provenance"]["request_max_tokens"] == 4096, (
        "the stale 4000-ceiling truncation survived a retry at 4096; the published "
        "record is then an attempt made under a ceiling known to be wrong"
    )


def test_a_failed_retry_at_the_SAME_ceiling_does_not_replace():
    """An equal ceiling buys a different sample, not better evidence."""
    existing = {"t1": _truncated("t1", 4096, primary=0.0)}
    fresh = {"t1": _truncated("t1", 4096, primary=0.0)}
    merged = _merge(existing, fresh, inc_tr=True)
    assert merged["t1"] is existing["t1"], "a same-ceiling re-roll must not replace"


def test_a_fresh_error_still_never_downgrades_an_existing_OK():
    """The load-bearing surrounding rule must survive the fix."""
    existing = {"t1": _clean("t1", 4000, primary=1.0)}
    fresh = {"t1": _truncated("t1", 4096)}
    merged = _merge(existing, fresh, inc_tr=True)
    assert merged["t1"]["score"]["primary"] == 1.0, (
        "a fresh truncation replaced a clean existing answer"
    )


def test_a_fresh_OK_still_replaces_a_truncation():
    existing = {"t1": _truncated("t1", 4000)}
    fresh = {"t1": _clean("t1", 4096, primary=0.85)}
    merged = _merge(existing, fresh, inc_tr=True)
    assert merged["t1"]["score"]["primary"] == 0.85


def test_the_replacement_does_not_fire_without_include_truncated():
    """Without the flag, the NEW branch must not fire.

    Note what this cannot be tested with. `_record_is_ok(..., include_truncated=False)`
    treats a truncated-but-scored record as a REAL VERDICT (cli.py:216), so a plain
    truncated fresh record takes the ordinary "a fresh OK replaces" path and lands in
    `merged` for a reason that has nothing to do with the ceiling. My first draft of
    this test asserted it would not, and the test was wrong rather than the code --
    checked against `_record_is_ok`'s own docstring rather than assumed.

    So the fresh record here carries an ERROR breakdown, which is never OK under
    either flag. That isolates the new branch: it must not fire, because `inc_tr`
    is False.
    """
    existing = {"t1": _truncated("t1", 4000)}
    errored = _truncated("t1", 4096)
    errored["score"]["breakdown"] = {"error": "ValueError: no parseable JSON"}
    merged = _merge(existing, {"t1": errored}, inc_tr=False)
    assert merged["t1"]["provenance"]["request_max_tokens"] == 4000, (
        "the new higher-ceiling branch fired without --include-truncated"
    )


def test_the_replacement_DOES_fire_for_an_errored_retry_at_a_higher_ceiling():
    """The real Cohere case: the fresh attempt errored (unparseable truncated JSON).

    All 5 grant truncations look like this -- `finish_reason == "length"` AND an
    error breakdown, because truncated JSON does not parse. If the branch only
    handled scored truncations it would not have touched a single real record.
    """
    existing = {"t1": _truncated("t1", 4000)}
    existing["t1"]["score"]["breakdown"] = {"error": "ValueError: no parseable JSON"}
    errored = _truncated("t1", 4096)
    errored["score"]["breakdown"] = {"error": "ValueError: no parseable JSON"}
    merged = _merge(existing, {"t1": errored}, inc_tr=True)
    assert merged["t1"]["provenance"]["request_max_tokens"] == 4096


def test_an_unknown_fresh_ceiling_does_not_replace():
    """A pre-v9 fresh record cannot prove it was an improvement, so it must not win."""
    existing = {"t1": _truncated("t1", 4000)}
    fresh = {"t1": _truncated("t1", None)}
    merged = _merge(existing, fresh, inc_tr=True)
    assert merged["t1"]["provenance"]["request_max_tokens"] == 4000


def test_the_branch_is_actually_IN_cmd_retry_failed_not_only_in_this_test():
    """The guard on this test file itself.

    `_merge` above re-implements the merge loop so the branch logic can be driven
    directly. That is a probe that reimplements its caller, and on its own it would
    keep passing if someone deleted the branch from `_cmd_retry_failed` while
    leaving `_record_ceiling` in place -- the tests would then be measuring this
    file, not the shipped code. Measured 2026-09-12: against the pre-fix cli.py the
    suite above failed 5 of 8, but by AttributeError on the missing helper rather
    than on the missing branch, which is the wrong reason.

    So assert the real call site carries it.
    """
    src = (REPO / "framework" / "cli.py").read_text(encoding="utf-8")
    body = src[src.index("def _cmd_retry_failed"):]
    body = body[: body.index("\ndef ")] if "\ndef " in body else body
    assert "_record_ceiling(rec)" in body, (
        "_cmd_retry_failed no longer consults the record's ceiling; the merge has "
        "regressed to discarding a retry made at a corrected ceiling"
    )
    assert "_record_is_truncated(merged[tid])" in body, (
        "the higher-ceiling replacement branch is gone from _cmd_retry_failed"
    )
