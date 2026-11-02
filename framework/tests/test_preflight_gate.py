"""framework.preflight: the gate before a costly run (owner request 2026-11-01).

A run larger than the pretest, with a player that calls a model, is refused before anything is
spent unless the run-shaping files are committed and each configuration has a stored "go" review.
Every record of a model-calling player must carry its evidence; a full run stops at the first that
does not (after writing it), a pretest-sized run reports them all at the end.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml

from framework import cli_session_log, hermetic, preflight
from framework.storage import read_records
from framework.tests._container_fakes import in_process_runner  # noqa: F401  (fixture)

GRIM = Path("arenas/grim-consistency-v1")
REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _own_stores(monkeypatch, tmp_path_factory):
    """Every test gets fresh run-env and call-ledger stores. The private root conftest already does
    this; the public mirror ships no root conftest, and there a second run of this file failed
    test_a_costly_run_without_a_review_is_refused_before_any_call on the review the first run
    stored (measured in a staged mirror, 2026-11-02)."""
    monkeypatch.setenv("SCIENCEARENA_RUN_ENV_DIR", str(tmp_path_factory.mktemp("run_env")))
    monkeypatch.setenv("SCIENCEARENA_CALL_CUSTODY_DIR", str(tmp_path_factory.mktemp("call_custody")))


@pytest.fixture
def fake_claude(monkeypatch, tmp_path):
    """A Claude CLI that answers, and a session log that says what effort it sent."""
    monkeypatch.setattr(hermetic, "cli_version", lambda b: "9.9.9-test")
    state = {"effort_logged": "high", "calls": 0}
    log = tmp_path / "session.jsonl"
    log.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cli_session_log, "find_claude_transcript", lambda sid: log)
    monkeypatch.setattr(cli_session_log, "extract_claude", lambda p: {
        "effort_sent": state["effort_logged"], "harness_version": "9.9.9-test",
        "models_billed": ["claude-sonnet-x"], "model_requested": "claude-sonnet-x",
        "models_served": ["claude-sonnet-x"], "tools_called": []})
    import shutil
    real_which, real_run = shutil.which, subprocess.run
    monkeypatch.setattr(shutil, "which", lambda c, *a, **k: real_which(c, *a, **k) or (f"/usr/bin/{c}" if c == "claude" else None))

    def fake_run(argv, *a, **kw):
        if isinstance(argv, list) and argv and "claude" in str(argv[0]).lower():
            if argv[1:] == ["--version"]:
                return subprocess.CompletedProcess(argv, 0, stdout="9.9.9-test", stderr="")
            state["calls"] += 1
            return subprocess.CompletedProcess(argv, 0, stderr="", stdout=json.dumps(
                {"type": "result", "result": json.dumps({"records": []}), "session_id": "x", "usage": {}}))
        return real_run(argv, *a, **kw)
    monkeypatch.setattr(subprocess, "run", fake_run)
    return state


def _registry(tmp_path) -> Path:
    entry = {"player_id": "gate-probe", "player_type": "ai-model", "player_version": "claude-sonnet-x",
             "adapter_class": "SubprocessCliAdapter", "cli_command": ["claude", "--print", "--model", "claude-sonnet-x"],
             "cli_json_envelope": "claude", "cli_effort": "high",
             "prompt_template_path": "players/prompts/grim_consistency.txt",
             "confidence_strategy": "native", "deterministic": True}
    reg = tmp_path / "registry.yaml"
    reg.write_text(yaml.safe_dump([entry]), encoding="utf-8")
    return reg


def _run(tmp_path, *, max_tasks):
    from framework.runner import run_tournament
    return run_tournament(arena_dir=GRIM, task_set_version="v1", registry_path=_registry(tmp_path),
                          player_ids=["gate-probe"], output_path=tmp_path / "run.jsonl", trials=1,
                          timeout_s=60, seed=0, split="revealed", max_tasks=max_tasks, public_only=True)


def test_a_costly_run_without_a_review_is_refused_before_any_call(tmp_path, fake_claude, in_process_runner,
                                                                    monkeypatch):
    monkeypatch.setattr(preflight, "dirty_run_files", lambda arena_dir: [])
    with pytest.raises(preflight.PreflightRefused, match="no stored 'go' review"):
        _run(tmp_path, max_tasks=preflight.PRETEST_N + 1)
    assert fake_claude["calls"] == 0


def test_a_costly_run_from_a_dirty_tree_is_refused(tmp_path, fake_claude, in_process_runner, monkeypatch):
    monkeypatch.setattr(preflight, "dirty_run_files", lambda arena_dir: ["?? players/prompts/new.txt"])
    with pytest.raises(preflight.PreflightRefused, match="untracked files shape this run"):
        _run(tmp_path, max_tasks=None)
    assert fake_claude["calls"] == 0


def test_a_pretest_needs_no_review_and_its_digest_unlocks_the_full_run(tmp_path, fake_claude, in_process_runner,
                                                                        monkeypatch, caplog):
    import logging
    monkeypatch.setattr(preflight, "dirty_run_files", lambda arena_dir: [])
    caplog.set_level(logging.INFO, logger="framework.runner")
    assert _run(tmp_path, max_tasks=preflight.PRETEST_N) == preflight.PRETEST_N
    digest = next(r.getMessage().split(" = ")[1].split(" ")[0] for r in caplog.records
                  if r.getMessage().startswith("configuration gate-probe = "))
    rnd = tmp_path / "round"
    rnd.mkdir()
    (rnd / "ROUND.json").write_text(json.dumps({"complete": True, "seats": [{"seat": "sonnet", "verdict": "OK"}]}),
                                    encoding="utf-8")
    path = preflight.write_review(digests={digest: "gate-probe"}, verdict="go", seats=["opus-session", "sonnet"],
                                  consult_dir=str(rnd), pretest_run_files=["run.jsonl"])
    assert json.loads(path.read_text(encoding="utf-8"))["digests"] == {digest: "gate-probe"}
    (tmp_path / "run.jsonl").unlink()
    assert _run(tmp_path, max_tasks=preflight.PRETEST_N + 1) == preflight.PRETEST_N + 1


def test_a_full_run_stops_at_the_first_record_without_its_evidence(tmp_path, fake_claude, in_process_runner,
                                                                    monkeypatch):
    monkeypatch.setattr(preflight, "dirty_run_files", lambda arena_dir: [])
    monkeypatch.setattr(preflight, "find_review", lambda dg: {"_path": "test-review"})
    fake_claude["effort_logged"] = "medium"            # asked high, the CLI's log says medium
    with pytest.raises(preflight.RecordContractViolation, match="effort_applied=False"):
        _run(tmp_path, max_tasks=None)
    assert fake_claude["calls"] == 1                   # stopped after the first call ...
    assert len(read_records(tmp_path / "run.jsonl")) == 1   # ... and its record was kept


def test_a_pretest_reports_every_violation_at_the_end(tmp_path, fake_claude, in_process_runner):
    fake_claude["effort_logged"] = "medium"
    with pytest.raises(preflight.RecordContractViolation, match=r"3 record\(s\) lack their evidence"):
        _run(tmp_path, max_tasks=preflight.PRETEST_N)
    assert len(read_records(tmp_path / "run.jsonl")) == preflight.PRETEST_N


def test_record_problems_reads_every_call_of_an_aggregate():
    entry = {"adapter_class": "RegcheckShimAdapter", "regcheck_client": "openai"}
    ok = {"checks": {"effort_applied": True, "session_log": "found"}, "launch": {"effort_requested": "low"}}
    bad = {"checks": {"effort_applied": False, "session_log": "not-found"}, "launch": {"effort_requested": "low"}}
    rec = {"env_sha": "e", "call_config_sha256": "c", "model_call": {"recorder_version": 1, "blocks": [ok, bad]}}
    probs = preflight.record_problems(rec, entry)
    assert any(p.startswith("call 2: effort_applied=False") for p in probs)
    assert any("session_log='not-found'" in p for p in probs)
    assert preflight.record_problems(rec, {"adapter_class": "RCliAdapter"}) == []      # a tool is not gated


def test_untracked_files_count_but_the_arenas_own_run_files_do_not(tmp_path):
    root = tmp_path / "repo"
    (root / "arenas" / "a" / "runs").mkdir(parents=True)
    (root / "players").mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    (root / "arenas" / "a" / "runs" / "x.jsonl").write_text("{}", encoding="utf-8")
    (root / "players" / "new_prompt.txt").write_text("x", encoding="utf-8")
    dirty = preflight.dirty_run_files(root / "arenas" / "a", repo_root=root)
    assert dirty == ["?? players/new_prompt.txt"]


def test_record_problems_requires_the_served_model_and_refuses_unallowed_tool_calls():
    """Go review 2026-11-01 (Sol): a missing billing check passed, so a Codex record with no evidence of
    which model answered was accepted; and a call that used a tool it was not allowed was no violation."""
    entry = {"adapter_class": "RegcheckShimAdapter", "regcheck_client": "openai"}

    def rec(checks, allow=()):
        b = {"checks": {"effort_applied": True, "session_log": "found", **checks},
             "launch": {"effort_requested": "low", "allow_tools": list(allow)}}
        return {"env_sha": "e", "call_config_sha256": "c", "model_call": {"recorder_version": 1, "blocks": [b]}}

    assert preflight.record_problems(rec({"only_requested_model_served": True, "tools_called": 0}), entry) == []
    assert any("answered" in p for p in preflight.record_problems(rec({"tools_called": 0}), entry))
    assert any("answered" in p for p in preflight.record_problems(rec({"only_requested_model_served": False}), entry))
    assert any("tool" in p for p in preflight.record_problems(
        rec({"only_requested_model_served": True, "tools_called": 2}), entry))
    assert preflight.record_problems(rec({"only_requested_model_served": True, "tools_called": 2}, allow=["Read"]),
                                     entry) == []


def test_code_closure_follows_imports_and_sees_a_change_in_an_imported_file(tmp_path):
    (tmp_path / "framework").mkdir()
    a = tmp_path / "framework" / "a.py"
    a.write_text("from framework import b\nimport requests\n\ndef f():\n    from framework.c import x\n", encoding="utf-8")
    (tmp_path / "framework" / "b.py").write_text("X = 1\n", encoding="utf-8")
    (tmp_path / "framework" / "c.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "framework" / "unrelated.py").write_text("y = 1\n", encoding="utf-8")
    files, external = preflight.code_closure([a], root=tmp_path)
    assert set(files) == {"framework/a.py", "framework/b.py", "framework/c.py"}   # a lazy import counts too
    assert "requests" in external
    (tmp_path / "framework" / "b.py").write_text("X = 2\n", encoding="utf-8")
    assert preflight.code_closure([a], root=tmp_path)[0] != files


@pytest.mark.skipif(not (REPO / "arenas" / "power-review-code-v1" / "scorer.py").is_file(),
                    reason="the power-review arenas are withheld from the public mirror -- NOT verified here")
def test_the_digest_covers_the_recorder_and_the_code_the_player_imports(monkeypatch):
    """Go review 2026-11-01 (Sol): _module_file passed type(fn) for a function, so recorder_code_sha256 was
    None in every stored configuration; the adapter's base class, the scorer's shared modules and outside
    packages were not covered either, so a committed change to any of them kept a reviewed digest."""
    from framework import model_call
    from players.adapters.power_review_cli import PowerReviewCliAdapter
    monkeypatch.setattr(hermetic, "cli_version", lambda b: "9.9.9-test")
    reg = yaml.safe_load((REPO / "players" / "power_review_registry.yaml").read_text(encoding="utf-8"))
    entries = reg["players"] if isinstance(reg, dict) else reg
    entry = next(e for e in entries if e["player_id"] == "claude-haiku-5-5-power-review")
    adapter = object.__new__(PowerReviewCliAdapter)
    comps = preflight.config_components(entry, adapter, arena_id="power-review-code-v1", task_set_version="v4",
                                        env_sha="e", scorer_path=REPO / "arenas" / "power-review-code-v1" / "scorer.py")
    assert comps["recorder_code_sha256"] == preflight._sha_file(Path(model_call.__file__))
    for f in ("framework/player_adapter.py", "framework/cli_session_log.py", "framework/hermetic.py",
              "framework/model_call.py", "arenas/power_review_family.py", "players/adapters/power_review_free.py",
              "players/adapters/power_review_cli.py", "arenas/power-review-code-v1/scorer.py"):
        assert f in comps["code_sha256"], f
    assert "powerbench" in comps["external_packages"]
