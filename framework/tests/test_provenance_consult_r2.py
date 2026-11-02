"""Regressions from the second-model review of the provenance work (Sonnet seat, 2026-11-01).

Round: docs/consults/2026-11-01-run-provenance-r2/round/01-sonnet.md. Each test below reproduced
its finding against commit a0fb1c88 (seen failing) before the fix.
"""
from __future__ import annotations

import datetime as dt

import pytest

from framework import cli_session_log as L
from framework import hermetic, model_call, preflight
from framework import publication_contract as pc

CLI_CFG = {"cli": {"api_type": "cli:codex"}}
V13_CLI = {"schema_version": 13, "player_type": "ai-model", "env_sha": "e", "call_config_sha256": "cli"}
CLI_ENTRY = {"player_id": "p", "adapter_class": sorted(hermetic.CLI_ADAPTER_CLASSES)[0]}


@pytest.fixture(autouse=True)
def _no_real_cli(monkeypatch):
    import shutil
    real = shutil.which
    monkeypatch.setattr(shutil, "which",
                        lambda c, *a, **k: real(c, *a, **k) or (f"/usr/bin/{c}" if c in ("claude", "codex") else None))
    monkeypatch.setattr(hermetic, "cli_version", lambda b: "2.1.316 (Claude Code)")
    monkeypatch.setattr(model_call, "codex_supported_efforts",
                        lambda m: (("low", "medium", "high"), "medium", "test catalog"))


# Finding 1: an aggregate with no blocks read as "requested and confirmed".
@pytest.mark.parametrize("mc", [{"blocks": []}, {}, {"recorder_version": 1}])
def test_a_model_call_with_no_call_in_it_is_not_evidence(mc):
    rec = {**V13_CLI, "model_call": mc}
    assert pc.record_problem(rec, frozen_utc="2026-11-01T00:00:00Z", call_config_of=CLI_CFG.get) is not None
    env = {**rec, "env_sha": "e"}
    assert preflight.record_problems(env, CLI_ENTRY) != []


# Finding 2: a CLI version change was a substring test and never stopped anything.
def test_a_version_that_merely_shares_a_prefix_is_a_change(monkeypatch, tmp_path):
    plan = model_call.prepare(["claude", "--print", "--model", "claude-x"], effort="low", purpose="t")
    (tmp_path / "x.jsonl").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(L, "find_claude_transcript", lambda sid: tmp_path / "x.jsonl")
    monkeypatch.setattr(L, "extract_claude", lambda p: {"effort_sent": "low", "harness_version": "2.1.3",
                                                         "models_billed": [], "tools_called": []})
    block = model_call.finish(plan, returncode=0, stdout="", stderr="")
    assert block["checks"]["cli_version_changed"] == {"launch": "2.1.316", "session_log": "2.1.3"}


def test_a_cli_version_change_is_a_contract_problem():
    rec = {**V13_CLI, "model_call": {"launch": {"effort_requested": "low"},
                                     "checks": {"effort_applied": True, "session_log": "found",
                                                "cli_version_changed": {"launch": "2.1.316", "session_log": "2.1.320"}}}}
    assert any("version" in p for p in preflight.record_problems(rec, CLI_ENTRY))


# Finding 3: trials multiplied the calls of a "pretest".
def test_trials_count_toward_the_pretest_size():
    assert preflight.is_costly(3, None, trials=30) is True
    assert preflight.is_costly(3, None, trials=1) is False
    assert preflight.is_costly(None, {"a", "b"}, trials=2) is True


def test_a_go_review_must_point_at_a_real_consult_round(tmp_path, monkeypatch):
    monkeypatch.setenv("SCIENCEARENA_RUN_ENV_DIR", str(tmp_path / "store"))
    with pytest.raises(ValueError, match="ROUND.json"):
        preflight.write_review(digests={"d" * 64: "p"}, verdict="go", seats=["opus", "sonnet"],
                               consult_dir=str(tmp_path / "no-such-round"), pretest_run_files=[])


# Finding 4: the Python copy read 13.0 as legacy while the site read it as v13.
def test_an_integral_float_schema_version_is_that_version():
    rec = {**V13_CLI, "schema_version": 13.0, "model_call": None}
    assert pc.record_problem(rec, frozen_utc="2026-11-01T00:00:00Z", call_config_of=CLI_CFG.get) == \
        "v13 command-line-tool record without its model_call block"


# Finding 6: a Codex stored thread setting is not the request log.
def test_a_codex_record_backed_only_by_the_thread_table_is_unverified(monkeypatch):
    plan = model_call.prepare(["codex", "exec", "-m", "gpt-x"], effort="high", purpose="t")
    monkeypatch.setattr(model_call, "_find_codex_thread",
                        lambda p, e: ({"id": "t1", "rollout_path": "gone"}, [], "found"))
    monkeypatch.setattr(L, "codex_rollout_file", lambda p: None)
    monkeypatch.setattr(L, "extract_codex_db", lambda t, turns: {"effort_sent": "high", "turns": 1,
                                                                  "reasoning_items": 0, "source": "codex-state-db"})
    block = model_call.finish(plan, returncode=0, stdout="", stderr="")
    assert block["checks"]["effort_applied"] == "unverified"
    assert block["checks"]["session_log"] == "thread-table-only"


# Finding 7: a thread already claimed by another call must not be matched again.
def test_a_codex_thread_is_claimed_once(monkeypatch, tmp_path):
    now = dt.datetime.now(dt.timezone.utc)
    plan = model_call.prepare(["codex", "exec", "-m", "gpt-x"], effort="low", purpose="t", cwd=str(tmp_path))
    thread = {"id": "shared", "cwd": str(plan.spawn["cwd"]), "created_at_ms": now.timestamp() * 1000,
              "rollout_path": "r"}
    monkeypatch.setattr(L, "codex_threads", lambda cwd_like: [thread])
    monkeypatch.setattr(L, "codex_thread_turns", lambda ids: {})
    first = model_call._find_codex_thread(plan, now)
    second = model_call._find_codex_thread(plan, now)
    assert first[2] == "found" and second[2] == "not-found"


# Finding 9: the held-out reduction kept the Claude session id inside argv.
def test_held_out_reduction_masks_the_session_id_in_argv():
    block = {"recorder_version": 1, "session_id": "abc-123",
             "launch": {"argv": ["claude", "--session-id", "abc-123", "--print"]}, "checks": {}}
    out = model_call.public_block(block, held_out=True)
    assert "abc-123" not in str(out)


# Premise attack: `via_recorder=True` is an ordinary keyword; only the recorder may pass it.
def test_only_the_recorder_claims_to_be_the_recorder():
    repo = model_call.Path(__file__).resolve().parents[2]
    offenders = []
    for f in repo.rglob("*.py"):
        rel = f.relative_to(repo).as_posix()
        if "/tests/" in f"/{rel}" or rel.startswith((".claude/", "node_modules/", "leaderboard-app/")):
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "via_recorder=True" in text and rel != "framework/model_call.py":
            offenders.append(rel)
    assert offenders == []


# Finding 5 (digest gap): a containerised player's digest names the adapter inside the image.
def test_the_digest_hashes_the_adapter_inside_the_container():
    from framework import container
    entry = {"player_id": "p", "adapter_class": "SubprocessCliAdapter", "cli_command": ["claude"]}
    wrapper = object.__new__(container.ContainerizedAdapter)
    comps = preflight.config_components(entry, wrapper, arena_id="a", task_set_version="v1", env_sha="e",
                                        scorer_path=None)
    assert comps["adapter_class"] == "SubprocessCliAdapter"
    assert comps["wrapper_class"] == "ContainerizedAdapter"
