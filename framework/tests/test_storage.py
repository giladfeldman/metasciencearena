"""Tests for run-record JSONL storage."""
import json
import pytest
from framework.storage import RunRecordWriter, read_records, RunRecordValidationError


def _good_record():
    return {
        "run_id": "r-1", "arena_id": "fake-arena", "task_set_version": "v1", "task_id": "t1",
        "player_id": "stub-pass", "player_version": "0.0.1", "player_type": "tool",
        "input_hash": "0" * 64, "output": {"label": "ok"},
        "score": {"primary": 1.0, "breakdown": {}},
        "timestamp_utc": "2026-04-29T12:00:00Z",
    }


def test_writer_appends_valid_records(tmp_path):
    out = tmp_path / "runs.jsonl"
    with RunRecordWriter(out) as w:
        w.append(_good_record())
        w.append({**_good_record(), "run_id": "r-2"})
    records = read_records(out)
    assert len(records) == 2
    assert records[0]["run_id"] == "r-1"


def test_writer_rejects_invalid_record(tmp_path):
    out = tmp_path / "runs.jsonl"
    with RunRecordWriter(out) as w:
        with pytest.raises(RunRecordValidationError):
            w.append({"run_id": "r-1"})  # missing required fields


def test_read_records_handles_empty_file(tmp_path):
    out = tmp_path / "empty.jsonl"
    out.write_text("", encoding="utf-8")
    assert read_records(out) == []


def test_read_records_survives_a_torn_trailing_line(tmp_path, caplog):
    """A killed run can leave ONE partially written record.

    Observed 2026-08-03: an interrupted tournament left a fragment that made the
    whole 12-record file unparseable, so a single torn write destroyed an entire
    tournament's results. The fragment is unrecoverable, but it must not take
    the valid records with it — and it must be logged, not silently dropped.
    """
    out = tmp_path / "torn.jsonl"
    good = json.dumps(_good_record())
    out.write_text(good + "\n" + good[:40], encoding="utf-8")  # truncated 2nd line

    with caplog.at_level("WARNING"):
        records = read_records(out)

    assert len(records) == 1, "the intact record must survive"
    assert records[0]["run_id"] == "r-1"
    assert any("unparseable run record" in m for m in caplog.messages)


def test_iter_records_also_survives_a_torn_line(tmp_path):
    from framework.storage import iter_records
    out = tmp_path / "torn2.jsonl"
    good = json.dumps(_good_record())
    out.write_text(good[:30] + "\n" + good + "\n", encoding="utf-8")  # torn FIRST
    assert [r["run_id"] for r in iter_records(out)] == ["r-1"]


def test_append_mode_doubles_on_rerun(tmp_path):
    # DR-0013: documents the default-append footgun the overwrite flag fixes.
    out = tmp_path / "runs.jsonl"
    for _ in range(2):
        with RunRecordWriter(out) as w:
            w.append(_good_record())
    assert len(read_records(out)) == 2  # doubled


def test_overwrite_replaces_on_rerun(tmp_path):
    out = tmp_path / "runs.jsonl"
    for _ in range(2):
        with RunRecordWriter(out, overwrite=True) as w:
            w.append(_good_record())
    assert len(read_records(out)) == 1  # replaced, not doubled


def test_append_on_nonempty_warns(tmp_path, caplog):
    import logging
    out = tmp_path / "runs.jsonl"
    with RunRecordWriter(out) as w:
        w.append(_good_record())
    with caplog.at_level(logging.WARNING, logger="framework.storage"):
        with RunRecordWriter(out) as w:  # append onto a non-empty file
            w.append(_good_record())
    assert any("non-empty file" in r.message for r in caplog.records)
    # overwrite mode must NOT warn (it replaces cleanly).
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="framework.storage"):
        with RunRecordWriter(out, overwrite=True) as w:
            w.append(_good_record())
    assert not any("non-empty file" in r.message for r in caplog.records)
