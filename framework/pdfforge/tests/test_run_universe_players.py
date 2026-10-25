"""Driver status-file resolution (2026-10-18: `--status-file tmp/pu_runs/x.json` became
tmp/pu_runs/tmp/pu_runs/x.json and the driver crashed after its first job, hours in)."""
from pathlib import Path

import pytest

from framework.pdfforge.tools import run_universe_players as d


@pytest.fixture
def fake_repo(tmp_path, monkeypatch):
    # The real runs folder is gitignored (tmp/pu_runs) and absent on a fresh
    # checkout or worktree, where `_status_path` rightly refuses it (d-230f9f).
    # Point the driver at a throwaway repo that has one.
    runs = tmp_path / "tmp" / "pu_runs"
    runs.mkdir(parents=True)
    monkeypatch.setattr(d, "REPO", tmp_path)
    monkeypatch.setattr(d, "OUT", runs)
    return tmp_path


def test_bare_name_lives_in_the_runs_folder(fake_repo):
    assert d._status_path("status_X.json") == d.OUT / "status_X.json"


def test_a_path_is_taken_as_given_from_the_repo(fake_repo):
    assert d._status_path("tmp/pu_runs/status_X.json") == fake_repo / "tmp/pu_runs/status_X.json"


def test_a_missing_folder_fails_before_any_job_runs():
    with pytest.raises(SystemExit):
        d._status_path("no_such_dir/deeper/status_X.json")
