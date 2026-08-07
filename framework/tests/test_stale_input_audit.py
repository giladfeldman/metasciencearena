"""Input-freshness audit: a score must describe the task that exists TODAY.

Added 2026-08-04 (cycle 8) after a real, silent defect. `framework audit --symmetry`
only asks WHICH task_ids each player ran. It cannot see that a kept task_id's
CONTENT changed underneath it — which is exactly what in-place broadening does. So
a published number could describe text that no longer exists, and every existing
gate said PASS.

Found in production data: 86 records across 2 arenas, all predating the cycle-4/5
broadenings. The clearest case was statcheck on stats-extraction-v1 task
`t-tier6-d1-1-s0`, whose stored score 0.241 (4 extractions, 1 false positive) was
computed against pre-broaden text; re-running the SAME statcheck 1.5.0 against the
current text gives 0.429 (3 extractions, 0 false positives).
"""
from __future__ import annotations

from framework.audit import stale_input_records


def _rec(task_id, input_hash, player_id="p1"):
    return {"task_id": task_id, "input_hash": input_hash, "player_id": player_id}


def test_all_fresh_records_pass():
    records = [_rec("t1", "aaa"), _rec("t2", "bbb")]
    rep = stale_input_records(records, {"t1": "aaa", "t2": "bbb"}, arena_id="a")
    assert rep.ok
    assert rep.n_checked == 2
    assert rep.per_player == {}
    assert rep.problems == []


def test_changed_task_content_is_flagged():
    """The core regression: same task_id, different content, stale score."""
    records = [_rec("t1", "OLD_HASH")]
    rep = stale_input_records(records, {"t1": "NEW_HASH"}, arena_id="a")
    assert not rep.ok
    assert rep.per_player == {"p1": 1}
    assert "re-run" in rep.problems[0]


def test_flags_are_attributed_per_player():
    records = [
        _rec("t1", "OLD", player_id="statcheck"),
        _rec("t2", "OLD", player_id="statcheck"),
        _rec("t1", "NEW", player_id="fresh-player"),
    ]
    rep = stale_input_records(records, {"t1": "NEW", "t2": "NEW"}, arena_id="a")
    assert not rep.ok
    assert rep.per_player == {"statcheck": 2}
    assert "fresh-player" not in " ".join(rep.problems)


def test_task_ids_outside_the_current_set_are_ignored_not_flagged():
    """Private/held-out records and other task-set versions are out of scope.

    Flagging them would make the check cry wolf on every arena that runs both
    splits, and a noisy gate gets ignored — which is how the original defect
    survived.
    """
    records = [_rec("private-1", "zzz")]
    rep = stale_input_records(records, {"t1": "aaa"}, arena_id="a")
    assert rep.ok
    assert rep.n_checked == 0


def test_records_without_an_input_hash_are_skipped():
    """Older records predate the field; absence must not invent a failure."""
    records = [{"task_id": "t1", "player_id": "p1"}]
    rep = stale_input_records(records, {"t1": "aaa"}, arena_id="a")
    assert rep.ok
    assert rep.n_checked == 0


def test_empty_inputs_are_ok_but_check_nothing():
    rep = stale_input_records([], {}, arena_id="a")
    assert rep.ok
    assert rep.n_checked == 0


def test_current_input_hashes_handles_generators_without_a_split_param():
    """The PDF family + replication-target-lookup emit BOTH visibilities from one
    generate() call and take no `split` kwarg.

    Regression (2026-08-04, cycle 8): the first version of `current_input_hashes`
    always passed split=, so those 6 arenas raised TypeError and were reported
    [SKIP] — silently unchecked. They are the most real-paper-heavy arenas in the
    benchmark, and once the call was made signature-aware the check immediately
    found 84 more stale records (grobid-text-only 24; all 3
    replication-target-lookup players 20 each). A gate that cannot run on your
    biggest arenas is not a gate.
    """
    from pathlib import Path

    from framework.audit import current_input_hashes

    repo = Path(__file__).resolve().parents[2]
    for arena in ("pdf-section-structure-v1", "replication-target-lookup-v1"):
        hashes = current_input_hashes(repo / "arenas" / arena, "v1", split="revealed")
        assert hashes, f"{arena}: generator still not runnable in-process"
        assert all(isinstance(h, str) and h for h in hashes.values())


def _write_jsonl(path, records):
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def test_orphaned_retry_temp_with_unmerged_verdicts_is_reported(tmp_path):
    """A killed retry round leaves a temp; its OK verdicts are real, unmerged work.

    Regression (2026-08-04, cycle 8): this happened twice in one cycle. One orphan
    held 9 unmerged OK regcheck verdicts — silently lost work until noticed by
    hand. `retry-failed` does recover them, but only if someone knows to run it.
    """
    from framework.audit import orphaned_retry_temps

    runs = tmp_path / "runs" / "v1"
    _write_jsonl(runs / "p__revealed__c6.jsonl", [{"task_id": "t1"}])
    _write_jsonl(runs / "p__revealed__c6.retry-r1.jsonl",
                 [{"task_id": "t1"}, {"task_id": "t2"}, {"task_id": "t3"}])
    found = orphaned_retry_temps(tmp_path, "v1")
    assert len(found) == 1
    temp, n = found[0]
    assert temp.name == "p__revealed__c6.retry-r1.jsonl"
    assert n == 2, "t2 and t3 are unmerged; t1 is already in the target"


def test_fully_merged_retry_temp_is_not_reported(tmp_path):
    """An inert leftover is not lost work — do not cry wolf about it."""
    from framework.audit import orphaned_retry_temps

    runs = tmp_path / "runs" / "v1"
    _write_jsonl(runs / "p__revealed__c6.jsonl", [{"task_id": "t1"}, {"task_id": "t2"}])
    _write_jsonl(runs / "p__revealed__c6.retry-r1.jsonl", [{"task_id": "t1"}])
    assert orphaned_retry_temps(tmp_path, "v1") == []


def test_errored_records_in_a_retry_temp_are_not_counted_as_lost_work(tmp_path):
    from framework.audit import orphaned_retry_temps

    runs = tmp_path / "runs" / "v1"
    _write_jsonl(runs / "p__revealed__c6.jsonl", [{"task_id": "t1"}])
    _write_jsonl(runs / "p__revealed__c6.retry-r1.jsonl",
                 [{"task_id": "t9", "error": "429 rate limit"}])
    assert orphaned_retry_temps(tmp_path, "v1") == []


def test_empty_run_file_is_reported(tmp_path):
    """A 0-byte run file is a killed run, and it names a REAL player.

    Regression (2026-08-04, cycle 9): killing a stalled tournament left two 0-byte
    files (nemotron-nano-9b-siglang, gemma-4-31b-grim). `framework run` creates the
    target up front, so a run killed before its first task leaves a file the build
    can publish as a phantom zero-record player.
    """
    from framework.audit import empty_run_files

    runs = tmp_path / "runs" / "v1"
    runs.mkdir(parents=True)
    (runs / "real__revealed__c9.jsonl").write_text('{"task_id":"t1"}\n', encoding="utf-8")
    (runs / "phantom__revealed__c9.jsonl").write_text("", encoding="utf-8")
    found = empty_run_files(tmp_path, "v1")
    assert [p.name for p in found] == ["phantom__revealed__c9.jsonl"]


def test_no_empty_run_files_when_all_have_content(tmp_path):
    from framework.audit import empty_run_files

    runs = tmp_path / "runs" / "v1"
    runs.mkdir(parents=True)
    (runs / "a__revealed__c9.jsonl").write_text('{"task_id":"t1"}\n', encoding="utf-8")
    assert empty_run_files(tmp_path, "v1") == []


def test_current_input_hashes_returns_empty_for_a_missing_generator(tmp_path):
    """No generator => {} so the caller reports [SKIP], never a false all-clear."""
    from framework.audit import current_input_hashes

    assert current_input_hashes(tmp_path, "v1") == {}


def test_n_checked_reports_real_coverage():
    """n_checked must reflect what was actually compared, so a caller can tell
    'everything is fresh' apart from 'nothing was comparable'."""
    records = [_rec("t1", "aaa"), _rec("unknown", "bbb"), {"task_id": "t2"}]
    rep = stale_input_records(records, {"t1": "aaa", "t2": "ccc"}, arena_id="a")
    assert rep.n_checked == 1
    assert rep.ok
