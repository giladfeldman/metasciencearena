"""retry-failed: never re-sample an answer, keep retried raws findable, lose nothing on a crash.

Found 2026-10-15 finishing citation-claim-fidelity-v1 (Flash-Lite, 500 requests/day),
then hardened after the tier-3 consult of the same day (Sonnet + Sol):

1. Every errored record was re-queued, including ones whose call returned the
   model's answer (schema-rejected, or unparseable -- a bare RuntimeError with its
   raw response kept). Re-playing them re-samples the model until a format failure
   happens to parse (d-24a295). Retrying only calls that produced no answer is the
   DEFAULT since 2026-10-19; the old re-sampling needs `--resample-model-output-errors`...
2. ...and lets a fresh ANSWER displace a stale no-answer error (both seats: a 429
   retried into a schema-rejected answer kept the 429 and was re-sampled next round).
3. A round plays into `<target>.retry-rN.jsonl`, and the raw sidecar is named after
   the run file, so retried raws were left under the temp name.
4. The temp was deleted before the merged target was written (Sol #4), and a raw
   fold deleted its source even when a torn destination hid the appended members
   (Sol #5 / Sonnet #5).
"""
from __future__ import annotations

import gzip
import json
import types

import pytest

from framework import cli, run_env
from framework.paths import ARENAS_ROOT_ENV

ARENA = "prereg-deviation-v1"


def _rec(task_id, *, error=None, run_id=None):
    score = {"primary": 0.9, "breakdown": {}} if error is None else {
        "primary": 0.0, "breakdown": {"error": error}}
    return {"player_id": "p", "task_id": task_id, "run_id": run_id or f"run-{task_id}", "split": "revealed",
            "task_visibility": "public", "score": score}


def _write(path, recs):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")


def _raw_path(run_file):
    return run_env.raw_sidecar_path(run_file, held_out=False, arena_id=ARENA)


def _setup(monkeypatch, tmp_path, recs, universe, fresh_error=None):
    """fresh_error: task_id -> error string for the retried record (None = OK).
    Every fresh record gets a raw unless its error is a bare RuntimeError (no answer)."""
    fresh_error = fresh_error or {}
    target = tmp_path / "arenas" / ARENA / "runs" / "v1" / "p__revealed__r1.jsonl"
    _write(target, recs)
    monkeypatch.setenv(ARENAS_ROOT_ENV, str(tmp_path / "arenas"))
    monkeypatch.setattr(cli, "_resolve_seed", lambda *a, **k: 0)
    monkeypatch.setattr(cli, "_split_task_ids", lambda *a, **k: set(universe))
    played = []

    def fake_run(*, output_path, only_tasks, **kw):
        played.append(set(only_tasks))
        n = len(played)
        out = [_rec(t, error=fresh_error.get(t), run_id=f"fresh{n}-{t}") for t in sorted(only_tasks)]
        _write(output_path, out)
        for r in out:
            if r["score"]["breakdown"].get("error") != "RuntimeError":
                run_env.append_raw(_raw_path(output_path), {"run_id": r["run_id"], "raw_response": {}})
        return len(only_tasks)

    monkeypatch.setattr(cli, "run_tournament", fake_run)
    return target, played


def _args(**kw):
    base = dict(arena=ARENA, task_set="v1", player="p", tag="r1", split="revealed",
                seed=None, timeout=60, max_rounds=1, cooldown=0)
    base.update(kw)
    return types.SimpleNamespace(**base)


def _final(target):
    return {json.loads(l)["task_id"]: json.loads(l) for l in target.read_text().splitlines()}


RECS = [_rec("t1"), _rec("t2", error="RuntimeError"), _rec("t3", error="output_schema_violation"),
        _rec("t4", error="output_schema_violation: [] is not of type 'object'"),
        _rec("t6", error="RuntimeError")]


def test_with_the_flag_only_calls_that_got_no_answer_are_replayed(monkeypatch, tmp_path):
    target, played = _setup(monkeypatch, tmp_path, RECS, {"t1", "t2", "t3", "t4", "t5", "t6"})
    # t6 errored as a bare RuntimeError but its raw answer was kept: the model answered.
    run_env.append_raw(_raw_path(target), {"run_id": "run-t6", "raw_response": {}})
    assert cli._cmd_retry_failed(_args(keep_model_output_errors=True)) == 0
    assert played == [{"t2", "t5"}]
    final = _final(target)
    assert final["t3"]["score"]["breakdown"]["error"] == "output_schema_violation"
    assert final["t6"]["run_id"] == "run-t6"


def test_answer_preserving_is_the_default(monkeypatch, tmp_path):
    """d-24a295 (2026-10-19): with no flag at all, an answered task is never re-played."""
    target, played = _setup(monkeypatch, tmp_path, RECS, {"t1", "t2", "t3", "t4", "t5", "t6"})
    run_env.append_raw(_raw_path(target), {"run_id": "run-t6", "raw_response": {}})
    assert cli._cmd_retry_failed(_args()) == 0
    assert played == [{"t2", "t5"}]


def test_the_old_resampling_needs_an_explicit_flag(monkeypatch, tmp_path):
    """Control: the flag is what changes the queue, not some other edit."""
    _, played = _setup(monkeypatch, tmp_path, RECS, {"t1", "t2", "t3", "t4", "t6"})
    cli._cmd_retry_failed(_args(resample_model_output_errors=True))
    assert played == [{"t2", "t3", "t4", "t6"}]


def test_the_cli_parses_the_new_default_and_keeps_the_old_flag_accepted():
    parser = cli.build_parser()
    base = ["retry-failed", "--arena", ARENA, "--task-set", "v1", "--player", "p"]
    assert parser.parse_args(base).resample_model_output_errors is False
    assert parser.parse_args(base + ["--resample-model-output-errors"]).resample_model_output_errors is True
    # Scripts and lessons written before 2026-10-19 pass the old flag: still accepted, now a no-op.
    assert parser.parse_args(base + ["--keep-model-output-errors"]).resample_model_output_errors is False
    with pytest.raises(SystemExit):
        parser.parse_args(base + ["--keep-model-output-errors", "--resample-model-output-errors"])


def test_orphan_recovery_defaults_to_answer_preserving(tmp_path):
    target = tmp_path / "arenas" / ARENA / "runs" / "v1" / "p__revealed__r1.jsonl"
    _write(target, [_rec("t2", error="RuntimeError")])
    _write(target.with_suffix(".retry-r1.jsonl"), [_rec("t2", error="output_schema_violation", run_id="f")])
    cli._recover_orphan_retry_temps(target)
    assert _final(target)["t2"]["run_id"] == "f"


def test_an_error_that_merely_mentions_the_marker_is_still_retried(monkeypatch, tmp_path):
    recs = [_rec("t1", error="RuntimeError: upstream said output_schema_violation")]
    _, played = _setup(monkeypatch, tmp_path, recs, {"t1"})
    cli._cmd_retry_failed(_args(keep_model_output_errors=True))
    assert played == [{"t1"}]


def test_a_first_answer_displaces_a_quota_error_and_is_not_resampled(monkeypatch, tmp_path):
    """Both seats, 2026-10-15: quota error -> retried -> schema-rejected answer -> settled."""
    target, played = _setup(monkeypatch, tmp_path, [_rec("t1"), _rec("t2", error="RuntimeError")],
                            {"t1", "t2"}, fresh_error={"t2": "output_schema_violation"})
    cli._cmd_retry_failed(_args(keep_model_output_errors=True, max_rounds=3))
    assert played == [{"t2"}], "the answered task was re-sampled"
    assert _final(target)["t2"]["score"]["breakdown"]["error"] == "output_schema_violation"


def test_an_unanswered_retry_does_not_displace_anything(monkeypatch, tmp_path):
    target, played = _setup(monkeypatch, tmp_path, [_rec("t2", error="RuntimeError")], {"t2"},
                            fresh_error={"t2": "RuntimeError"})
    cli._cmd_retry_failed(_args(keep_model_output_errors=True, max_rounds=2))
    assert played == [{"t2"}, {"t2"}]
    assert _final(target)["t2"]["run_id"] == "run-t2"


def test_retried_raws_are_folded_into_the_run_files_sidecar(monkeypatch, tmp_path):
    target, _ = _setup(monkeypatch, tmp_path, RECS, {"t1", "t2", "t3", "t4"})
    run_env.append_raw(_raw_path(target), {"run_id": "run-t1", "raw_response": {}})
    cli._cmd_retry_failed(_args(keep_model_output_errors=True))
    raws = {r["run_id"] for r in run_env.read_raw(_raw_path(target))}
    assert raws == {"run-t1", "fresh1-t2"}
    assert not list(target.parent.glob("*.retry-r*")), "temp run file or temp raw sidecar left behind"
    assert _final(target)["t2"]["run_id"] in raws


def test_the_temp_survives_a_crash_before_the_merged_target_is_written(monkeypatch, tmp_path):
    """Sol #4: deleting the temp first lost the round's answers on a crash."""
    target, _ = _setup(monkeypatch, tmp_path, [_rec("t2", error="RuntimeError")], {"t2"})

    def boom(*a, **k):
        raise OSError("disk gone")

    monkeypatch.setattr(cli, "_write_records_atomically", boom)
    with pytest.raises(OSError):
        cli._cmd_retry_failed(_args(keep_model_output_errors=True))
    assert list(target.parent.glob("*.retry-r1.jsonl")), "the round's records were deleted before merging"
    assert _final(target)["t2"]["run_id"] == "run-t2"


def test_a_raw_fold_into_a_torn_sidecar_keeps_its_source(monkeypatch, tmp_path):
    """Sol #5: a torn member hides everything appended after it; the source must stay."""
    target, _ = _setup(monkeypatch, tmp_path, [_rec("t2", error="RuntimeError")], {"t2"})
    dst = _raw_path(target)
    run_env.append_raw(dst, {"run_id": "old", "raw_response": {}})
    torn = gzip.compress(json.dumps({"run_id": "torn"}).encode() + b"\n")[:-6]
    with dst.open("ab") as fh:
        fh.write(torn)
    cli._cmd_retry_failed(_args(keep_model_output_errors=True))
    tmp = target.with_suffix(".retry-r1.jsonl")
    assert tmp.exists() and _raw_path(tmp).exists(), "unreadable fold deleted its only copy"
    assert _final(target)["t2"]["run_id"] == "fresh1-t2"


def test_orphan_recovery_folds_the_orphans_raws_too(tmp_path):
    target = tmp_path / "arenas" / ARENA / "runs" / "v1" / "p__revealed__r1.jsonl"
    _write(target, [_rec("t1"), _rec("t2", error="RuntimeError")])
    temp = target.with_suffix(".retry-r1.jsonl")
    _write(temp, [_rec("t2", run_id="fresh-t2")])
    run_env.append_raw(_raw_path(temp), {"run_id": "fresh-t2", "raw_response": {}})
    cli._recover_orphan_retry_temps(target)
    assert {r["run_id"] for r in run_env.read_raw(_raw_path(target))} == {"fresh-t2"}
    assert not list(target.parent.glob("*.retry-r*"))


def test_orphan_recovery_lets_a_first_answer_displace_a_quota_error(tmp_path):
    target = tmp_path / "arenas" / ARENA / "runs" / "v1" / "p__revealed__r1.jsonl"
    _write(target, [_rec("t2", error="RuntimeError")])
    _write(target.with_suffix(".retry-r1.jsonl"), [_rec("t2", error="output_schema_violation", run_id="f")])
    cli._recover_orphan_retry_temps(target, keep_model_output_errors=True)
    assert _final(target)["t2"]["run_id"] == "f"
