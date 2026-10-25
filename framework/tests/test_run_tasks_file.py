"""`framework run --tasks-file`: play only the task_ids listed in a file.

Used to screen a candidate player on a known sub-population (e.g. the tasks that
hold every item of one mistake type) before paying for a full run. The subset is
forwarded to `run_tournament(only_tasks=...)`, the same seam `retry-failed` uses.
A listed id the run did not produce must fail loudly: a screen that silently
played fewer tasks than asked would read as a clean result on a smaller sample.
"""
from __future__ import annotations

import json

import pytest

from framework import cli
from framework.paths import ARENAS_ROOT_ENV


def _write_records(path, task_ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for t in task_ids:
            fh.write(json.dumps({"player_id": "p", "task_id": t}) + "\n")


@pytest.fixture
def stub_env(monkeypatch, tmp_path):
    (tmp_path / "arenas" / "a-v1").mkdir(parents=True)
    monkeypatch.setenv(ARENAS_ROOT_ENV, str(tmp_path / "arenas"))
    monkeypatch.setattr(cli, "_resolve_seed", lambda *a, **k: 0)
    seen = {}

    def fake_run(*, output_path, only_tasks, **kw):
        seen["only_tasks"] = only_tasks
        played = sorted(only_tasks) if only_tasks is not None else ["t1", "t2", "t3"]
        played = [t for t in played if t != "ghost"]  # an id the generator never emits
        _write_records(output_path, played)
        return len(played)

    monkeypatch.setattr(cli, "run_tournament", fake_run)
    return tmp_path, seen


def _argv(tmp_path, *extra):
    return ["run", "--arena", "a-v1", "--task-set", "v1", "--players", "p",
            "--split", "private", "--tag", "screen", "--overwrite", *extra]


def test_tasks_file_restricts_the_run_to_the_listed_ids(stub_env):
    tmp_path, seen = stub_env
    f = tmp_path / "ids.txt"
    f.write_text("# screen subset\nt2\n\nt3\n", encoding="utf-8")
    assert cli.main(_argv(tmp_path, "--tasks-file", str(f))) == 0
    assert seen["only_tasks"] == {"t2", "t3"}


def test_without_tasks_file_nothing_is_restricted(stub_env):
    tmp_path, seen = stub_env
    assert cli.main(_argv(tmp_path)) == 0
    assert seen["only_tasks"] is None


def test_a_listed_id_the_run_did_not_produce_fails(stub_env, capsys):
    tmp_path, _ = stub_env
    f = tmp_path / "ids.txt"
    f.write_text("t1\nghost\n", encoding="utf-8")
    assert cli.main(_argv(tmp_path, "--tasks-file", str(f))) != 0
    assert "1 of 2" in capsys.readouterr().err


def test_an_empty_tasks_file_is_refused(stub_env):
    tmp_path, seen = stub_env
    f = tmp_path / "ids.txt"
    f.write_text("# nothing\n\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        cli.main(_argv(tmp_path, "--tasks-file", str(f)))
    assert "only_tasks" not in seen


def test_a_record_from_an_earlier_run_does_not_count_as_played(monkeypatch, tmp_path):
    # RunRecordWriter APPENDS by default (framework/storage.py), so without
    # --overwrite the output file can already hold an earlier run's records. A listed
    # id this run did not produce must still fail, even though an old record carries
    # it (review finding 4, 2026-10-19).
    (tmp_path / "arenas" / "a-v1").mkdir(parents=True)
    monkeypatch.setenv(ARENAS_ROOT_ENV, str(tmp_path / "arenas"))
    monkeypatch.setattr(cli, "_resolve_seed", lambda *a, **k: 0)
    out = {}

    def fake_run(*, output_path, only_tasks, **kw):
        out["path"] = output_path
        played = [t for t in sorted(only_tasks) if t != "ghost"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as fh:
            for t in played:
                fh.write(json.dumps({"player_id": "p", "task_id": t}) + "\n")
        return len(played)

    monkeypatch.setattr(cli, "run_tournament", fake_run)
    argv = ["run", "--arena", "a-v1", "--task-set", "v1", "--players", "p",
            "--split", "private", "--tag", "screen"]
    f1 = tmp_path / "first.txt"
    f1.write_text("t1\n", encoding="utf-8")
    assert cli.main([*argv, "--tasks-file", str(f1)]) == 0
    with out["path"].open("a", encoding="utf-8") as fh:  # an earlier run that did play it
        fh.write(json.dumps({"player_id": "p", "task_id": "ghost"}) + "\n")
    f2 = tmp_path / "second.txt"
    f2.write_text("t2\nghost\n", encoding="utf-8")
    assert cli.main([*argv, "--tasks-file", str(f2)]) == 2
