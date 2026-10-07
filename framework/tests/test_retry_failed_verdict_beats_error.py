"""A fresh verdict must displace a stale transport error, even with no number.

WHY THIS FILE EXISTS (measured 2026-09-22). `framework retry-failed` merges a
replayed record back into a run file only when the fresh record is "OK", and
`_record_is_ok` requires a non-null `score.primary`. A task the SCORER EXCLUDES
carries `primary: None` by design, so such a record could never displace anything.

The consequence was published, not hypothetical. `rtl-v1-0017` in
`replication-target-lookup-v1` errored with a 429 on the first pass of the SciMeto
1.74.107 run, because that service rate-limits the lookup route to 10 requests per
60 s per IP. Its TRUE verdict is `excluded` (`label_status == 'unverifiable'`) --
exactly what the June 1.71.3 run recorded for the same task. On every retry the
task replayed SUCCESSFULLY (verified in isolation through the real adapter: 91
matches, 0.0 s) and the scorer excluded it, and the merge then threw that fresh
record away. FOURTEEN retry rounds across four invocations changed nothing, while
the file went on publishing `primary: 0.0` for it -- a zero manufactured by our own
rate limiting rather than by the player, which is the same class of defect as
scoring a model against a timeout we chose.

Two facets, both fixed and both pinned below:

1. the MERGE discarded a fresh verdict that carried no number;
2. the TODO QUEUE was keyed on the same predicate, so three permanently-excluded
   tasks were re-played in all 14 rounds and the queue reported "+0 newly OK" for
   ever -- a stall indistinguishable from a broken retry.

`_record_is_ok` is deliberately left alone: an exclusion is not a score, and the
"+N newly OK" counter must go on saying so.
"""
from __future__ import annotations

from framework.cli import _record_has_a_verdict, _record_is_ok


def _err(msg="HTTPError: 429 Client Error: Too Many Requests"):
    return {"task_id": "t", "score": {"primary": 0.0, "breakdown": {"error": msg}}}


def _excluded():
    return {"task_id": "t", "score": {"primary": None,
                                      "breakdown": {"excluded": True,
                                                    "reason": "label_status == 'unverifiable'"}}}


def _scored(v=1.0):
    return {"task_id": "t", "score": {"primary": v, "breakdown": {"f1": v}}}


def test_an_exclusion_is_a_verdict_even_though_it_is_not_a_score():
    """The distinction the whole fix rests on."""
    assert _record_has_a_verdict(_excluded()) is True
    assert _record_is_ok(_excluded()) is False


def test_a_transport_error_is_not_a_verdict():
    assert _record_has_a_verdict(_err()) is False
    assert _record_is_ok(_err()) is False


def test_a_scored_record_is_both():
    assert _record_has_a_verdict(_scored()) is True
    assert _record_is_ok(_scored()) is True


def test_a_zero_score_is_a_verdict_and_must_not_be_retried_away():
    """CONTROL, and the one that matters most. A genuine 0.0 looks numerically
    identical to the 0.0 a 429 record carries; only the error breakdown separates
    them. If this returned False, the fix would quietly re-roll every real failure
    a player earned -- which is the defect `retry-failed` was already caught
    committing once."""
    assert _record_has_a_verdict(_scored(0.0)) is True
    assert _record_is_ok(_scored(0.0)) is True


def test_the_two_predicates_actually_differ_somewhere():
    """VACUITY CONTROL. If `_record_has_a_verdict` were an alias for
    `_record_is_ok`, every assertion above would still pass except this one, and
    the bug would be back with a green suite."""
    disagreements = [r for r in (_err(), _excluded(), _scored(), _scored(0.0))
                     if _record_has_a_verdict(r) != _record_is_ok(r)]
    assert disagreements, "the two predicates never disagree -- the fix is a no-op"


def test_a_missing_breakdown_is_treated_as_a_verdict_not_an_error():
    """Defensive: a record shape with no breakdown at all must not be read as an
    error, which would put settled tasks back in the retry queue."""
    assert _record_has_a_verdict({"score": {"primary": 0.5}}) is True
    assert _record_has_a_verdict({"score": {}}) is True
    assert _record_has_a_verdict({}) is True


def test_the_published_run_file_has_no_error_records_left():
    """END TO END, against the file the defect was found in.

    `build-data.mjs` publishes this file. A 429 record here is a 0.0 on the board
    that measures our request pacing; the June 1.71.3 run has none, and after the
    fix neither does the 1.74.107 run. Two-sided: the June file is the control
    that a zero here can be real.
    """
    import json
    from framework.paths import arenas_root

    runs = arenas_root() / "replication-target-lookup-v1" / "runs" / "v1"
    for name in ("scimeto-replication-api__revealed__v1_71_3.jsonl",
                 "scimeto-replication-api-1-74-107__revealed__v1_74_107.jsonl"):
        path = runs / name
        if not path.is_file():
            continue
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                   if line.strip()]
        assert len(records) == 20, f"{name}: expected 20 records, got {len(records)}"
        errored = [r["task_id"] for r in records if not _record_has_a_verdict(r)]
        assert errored == [], f"{name} still carries transport errors: {errored}"
