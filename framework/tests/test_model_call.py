"""framework.model_call: every Claude / Codex CLI call states its effort and is recorded.

Background (2026-11-01). Three power-review arenas published "effort not recorded": the
players ran ``claude --print`` with no ``--effort``, so the level came from whatever the
CLI defaulted to that week, and nothing wrote it down. The recorder refuses such a call
before it is launched, gives every Claude call a session id the recorder chose (so its
transcript can be found and copied to custody), and checks afterwards, from the CLI's own
log, that the level asked for is the level that was sent.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

from framework import cli_session_log as L
from framework import hermetic, model_call

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _no_real_cli(monkeypatch):
    import shutil
    real = shutil.which
    monkeypatch.setattr(shutil, "which",
                        lambda c, *a, **k: real(c, *a, **k) or (f"/usr/bin/{c}" if c in ("claude", "codex") else None))
    monkeypatch.setattr(hermetic, "cli_version", lambda b: "0.0.0-test")
    monkeypatch.setattr(model_call, "codex_supported_efforts",
                        lambda m: (("low", "medium", "high"), "medium", "test catalog"))


# ---- refusals: a setting nobody could later state is never sent ---------------------------

def test_a_claude_call_without_an_effort_is_refused():
    with pytest.raises(model_call.UnrecordableCall, match="states no effort"):
        model_call.prepare(["claude", "--print", "--model", "sonnet"], effort=None, purpose="t")


def test_a_misspelt_effort_is_refused_not_silently_defaulted():
    # Claude Code 2.1.316 only warns on stderr for an unknown value and runs at its default.
    with pytest.raises(model_call.UnrecordableCall, match="not one of"):
        model_call.prepare(["claude", "--print", "--model", "claude-x"], effort="hgh", purpose="t")


def test_an_effort_the_codex_model_does_not_offer_is_refused():
    with pytest.raises(model_call.UnrecordableCall, match="not supported"):
        model_call.prepare(["codex", "exec", "-m", "gpt-x"], effort="ultra", purpose="t")


def test_an_effort_contradicting_the_argv_is_refused():
    with pytest.raises(model_call.UnrecordableCall, match="contradicts"):
        model_call.prepare(["claude", "--print", "--effort", "low"], effort="high", purpose="t")


def test_not_applicable_needs_a_reason():
    with pytest.raises(model_call.UnrecordableCall, match="needs effort_reason"):
        model_call.prepare(["claude", "--print", "--model", "claude-x"], effort=model_call.NOT_APPLICABLE, purpose="t")


def test_the_caller_cannot_choose_the_claude_session():
    with pytest.raises(model_call.UnrecordableCall, match="session id"):
        model_call.prepare(["claude", "--print", "--session-id", "x"], effort="low", purpose="t")


def test_a_scored_call_may_not_name_an_alias_or_rely_on_the_cli_default():
    for argv in (["claude", "--print", "--model", "opus"], ["claude", "--print"], ["codex", "exec"]):
        with pytest.raises(model_call.UnrecordableCall, match="alias|names no model"):
            model_call.prepare(argv, effort="low", purpose="t")
    plan = model_call.prepare(["claude", "--print", "--model", "opus"], effort="low", purpose="t", allow_alias=True)
    assert plan.launch["model_alias_allowed"] is True


def test_spawn_kwargs_refuses_claude_and_codex_outside_the_recorder():
    for argv in (["claude", "--print", "--model", "claude-x"], ["codex", "exec"]):
        with pytest.raises(hermetic.UnhardenedCliError, match="model_call"):
            hermetic.spawn_kwargs(argv)


# ---- what a prepared call carries ---------------------------------------------------------

def test_claude_gets_the_effort_and_a_session_id_and_the_launch_says_so():
    plan = model_call.prepare(["claude", "--print", "--model", "claude-sonnet-x"], effort="high", purpose="t")
    a = plan.argv
    assert a[a.index("--effort") + 1] == "high"
    assert a[a.index("--session-id") + 1] == plan.session_id
    assert plan.launch["effort_requested"] == "high"
    assert plan.launch["model_requested"] == "claude-sonnet-x"
    assert plan.launch["cli_version"] == "0.0.0-test"
    assert plan.launch["containment_profile"] == hermetic.PROFILE_VERSION


def test_an_effort_already_in_the_argv_is_kept_and_recorded():
    plan = model_call.prepare(["codex", "exec", "-m", "gpt-x", "-c", 'model_reasoning_effort="high"'],
                              effort=None, purpose="t")
    assert plan.launch["effort_requested"] == "high"
    assert plan.launch["effort_validated_against"] == "test catalog"
    assert plan.argv.count("-c") >= 1 and 'model_reasoning_effort=high' not in plan.argv


def test_each_codex_call_gets_its_own_folder_so_its_thread_can_be_found():
    p1 = model_call.prepare(["codex", "exec", "-m", "gpt-x"], effort="low", purpose="t")
    p2 = model_call.prepare(["codex", "exec", "-m", "gpt-x"], effort="low", purpose="t")
    assert p1.spawn["cwd"] != p2.spawn["cwd"]
    assert "model_reasoning_effort=low" in p1.argv


def test_a_prompt_passed_as_an_argument_is_recorded_as_a_digest():
    plan = model_call.prepare(["claude", "--print", "--model", "claude-x", "SECRET PAPER TEXT"], effort="low", purpose="t",
                              redact=("SECRET PAPER TEXT",))
    assert not any("SECRET" in a for a in plan.launch["argv"])
    assert any(a.startswith("<redacted sha256=") for a in plan.launch["argv"])


def test_the_claude_environment_is_an_allowlist(monkeypatch):
    # The desktop app hands its children ~123 variables, among them the account e-mail,
    # CLAUDE_EFFORT=xhigh and a messaging token. None of them is part of the instrument.
    for k, v in {"CLAUDE_CODE_USER_EMAIL": "x@y", "CLAUDE_EFFORT": "xhigh", "ANTHROPIC_BASE_URL": "http://x",
                 "SCIENCEARENA_PRIVATE_SEED": "s", "MCP_TOKEN": "t", "SOME_TOOL_HOME": "h"}.items():
        monkeypatch.setenv(k, v)
    plan = model_call.prepare(["claude", "--print", "--model", "claude-x"], effort="low", purpose="t")
    env = plan.spawn["env"]
    assert not {"CLAUDE_CODE_USER_EMAIL", "CLAUDE_EFFORT", "ANTHROPIC_BASE_URL", "SCIENCEARENA_PRIVATE_SEED",
                "MCP_TOKEN", "SOME_TOOL_HOME"} & set(env)
    assert set(env) <= hermetic.CLI_ENV_KEEP | set(hermetic.PINNED_ENV)
    assert plan.launch["env_var_names"] == sorted(env)
    assert plan.launch["env_pinned"] == {"DISABLE_AUTOUPDATER": "1"}


def test_caller_knobs_are_recorded_with_their_values():
    plan = model_call.prepare(["claude", "--print", "--model", "claude-x"], effort="low", purpose="t",
                              extra_env={"CLAUDE_CODE_MAX_OUTPUT_TOKENS": "64000"})
    assert plan.spawn["env"]["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "64000"
    assert plan.launch["env_pinned"]["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "64000"


# ---- after the call: the log is the evidence ----------------------------------------------

def _transcript(tmp_path, sid, effort):
    p = tmp_path / f"{sid}.jsonl"
    p.write_text(json.dumps({"type": "assistant", "sessionId": sid, "message": {
        "model": "claude-sonnet-x", "usage": {"input_tokens": 1, "output_tokens": 1}}}) + "\n", encoding="utf-8")
    return p


def test_finish_marks_effort_unapplied_when_the_cli_warned(monkeypatch):
    plan = model_call.prepare(["claude", "--print", "--model", "claude-x"], effort="low", purpose="t")
    monkeypatch.setattr(L, "find_claude_transcript", lambda sid: None)
    block = model_call.finish(plan, returncode=0, stdout="", stderr="Unknown --effort value 'low'")
    assert block["checks"]["effort_applied"] is False
    assert block["checks"]["session_log"] == "not-found"


@pytest.fixture
def custody_dir(monkeypatch, tmp_path):
    """The call-ledger folder for this test. The private root conftest already points it at a temp
    folder (autouse, so it is set before this runs); the public mirror ships no root conftest, so
    there the test sets its own (mirrored suite, 2026-11-02: KeyError without it)."""
    import os
    if not os.environ.get("SCIENCEARENA_CALL_CUSTODY_DIR"):
        monkeypatch.setenv("SCIENCEARENA_CALL_CUSTODY_DIR", str(tmp_path / "call_custody"))
    return Path(os.environ["SCIENCEARENA_CALL_CUSTODY_DIR"])


def test_finish_reads_the_effort_actually_sent_from_the_log(monkeypatch, tmp_path, custody_dir):
    plan = model_call.prepare(["claude", "--print", "--model", "claude-sonnet-x"], effort="high", purpose="t")
    path = _transcript(tmp_path, plan.session_id, "high")
    monkeypatch.setattr(L, "find_claude_transcript", lambda sid: path)
    monkeypatch.setattr(L, "extract_claude", lambda p: {"effort_sent": "medium", "models_billed": ["claude-sonnet-x"],
                                                         "model_requested": "claude-sonnet-x", "tools_called": []})
    block = model_call.finish(plan, returncode=0, stdout="", stderr="")
    assert block["checks"]["effort_applied"] is False      # asked high, the log says medium
    assert block["checks"]["only_requested_model_billed"] is True
    assert block["session_log"]["copied"] is True


def test_every_call_reaches_the_ledger_and_the_test_ledger_is_not_custody(monkeypatch, custody_dir):
    import os
    plan = model_call.prepare(["claude", "--print", "--model", "claude-x"], effort="low", purpose="ledger-probe")
    monkeypatch.setattr(L, "find_claude_transcript", lambda sid: None)
    model_call.finish(plan, returncode=0, stdout="", stderr="")
    root = custody_dir
    rows = [json.loads(x) for f in root.rglob("calls.jsonl") for x in f.read_text(encoding="utf-8").splitlines()]
    assert any(r["launch"]["cwd_purpose"] == "ledger-probe" for r in rows)
    art = os.environ.get("ARTICLE_REPOSITORY")
    if art:
        assert Path(art) not in root.parents


def test_the_raw_output_rides_on_the_block_but_never_reaches_the_record_or_ledger(monkeypatch, custody_dir):
    plan = model_call.prepare(["claude", "--print", "--model", "claude-x"], effort="low", purpose="raw-probe")
    monkeypatch.setattr(L, "find_claude_transcript", lambda sid: None)
    block = model_call.finish(plan, returncode=0, stdout="ANSWER TEXT", stderr="")
    assert block["_raw"]["stdout"] == "ANSWER TEXT"
    assert "_raw" not in model_call.public_block(block, held_out=False)
    agg = model_call.aggregate_blocks([block, block])
    assert all("_raw" not in b for b in model_call.public_block(agg, held_out=True)["blocks"])
    ledger = [f.read_text(encoding="utf-8") for f in custody_dir.rglob("calls.jsonl")]
    assert ledger and not any("ANSWER TEXT" in t for t in ledger)


def test_a_cli_update_between_launch_and_log_is_flagged(monkeypatch, tmp_path):
    plan = model_call.prepare(["claude", "--print", "--model", "claude-x"], effort="low", purpose="t")
    monkeypatch.setattr(L, "find_claude_transcript", lambda sid: tmp_path / "x.jsonl")
    (tmp_path / "x.jsonl").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(L, "extract_claude", lambda p: {"effort_sent": "low", "harness_version": "1.2.3",
                                                         "models_billed": [], "tools_called": []})
    block = model_call.finish(plan, returncode=0, stdout="", stderr="")
    assert block["checks"]["cli_version_changed"] == {"launch": "0.0.0-test", "session_log": "1.2.3"}


def test_run_keeps_the_block_when_the_call_times_out(monkeypatch):
    kept = []

    def boom(*a, **k):
        raise subprocess.TimeoutExpired(cmd="claude", timeout=1)
    monkeypatch.setattr(subprocess, "run", boom)
    monkeypatch.setattr(L, "find_claude_transcript", lambda sid: None)
    with pytest.raises(subprocess.TimeoutExpired):
        model_call.run(["claude", "--print", "--model", "claude-x"], effort="low", purpose="t", sink=kept.append)
    assert len(kept) == 1 and kept[0]["returncode"] is None


def test_a_held_out_block_drops_ids_timings_and_envelope():
    block = {"launch": {"effort_requested": "low"}, "session_id": "s", "session_log": {"copied": True},
             "envelope": {"usage": {}}, "started_utc": "t", "ended_utc": "t", "stderr_sha256": "x",
             "stderr_chars": 1, "facts": None, "checks": {"effort_applied": True}}
    pub = model_call.public_block(block, held_out=True)
    assert not {"session_id", "session_log", "envelope", "started_utc", "ended_utc"} & set(pub)
    assert pub["launch"]["effort_requested"] == "low" and pub["checks"]["effort_applied"] is True
    assert model_call.public_block(block, held_out=False)["session_id"] == "s"


# ---- the guard: nothing launches claude/codex around the recorder -------------------------

_LAUNCH = re.compile(r"""\[\s*["'](claude|codex)["']\s*,""")


def test_no_script_launches_claude_or_codex_without_the_recorder():
    """A static backstop for the runtime refusal in ``hermetic.spawn_kwargs``: a file that
    builds a ``["claude", ...]`` / ``["codex", ...]`` argv must import ``model_call``.
    Version probes (``--version``) and comments are exempt. It cannot see an argv built only
    from variables; those still meet the runtime refusal if they go through ``hermetic``."""
    files = subprocess.run(["git", "ls-files", "*.py"], cwd=REPO, capture_output=True, text=True,
                           check=True).stdout.split()
    offenders = []
    for rel in files:
        if "/tests/" in f"/{rel}" or rel.startswith("tests/") or rel in ("framework/model_call.py",
                                                                          "framework/hermetic.py"):
            continue
        text = (REPO / rel).read_text(encoding="utf-8", errors="replace")
        hits = [ln for ln in text.splitlines()
                if _LAUNCH.search(ln) and not ln.lstrip().startswith("#") and "--version" not in ln]
        if hits and "model_call" not in text:
            offenders.append((rel, hits[0].strip()[:100]))
    assert not offenders, offenders


def test_finish_checks_the_model_that_answered_not_only_the_bill(monkeypatch, tmp_path):
    """Go review 2026-11-01 (Sol): only the bill was compared with the request, and Codex has no bill, so a
    rerouted Codex answer passed. The served models are now compared for both CLIs."""
    plan = model_call.prepare(["claude", "--print", "--model", "claude-sonnet-x"], effort="high", purpose="t")
    (tmp_path / "x.jsonl").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(L, "find_claude_transcript", lambda sid: tmp_path / "x.jsonl")
    facts = {"effort_sent": "high", "models_billed": ["claude-sonnet-x"], "model_requested": "claude-sonnet-x",
             "tools_called": [], "models_served": ["claude-sonnet-x", "claude-opus-x"]}
    monkeypatch.setattr(L, "extract_claude", lambda p: dict(facts))
    assert model_call.finish(plan, returncode=0, stdout="", stderr="")["checks"]["only_requested_model_served"] is False
    facts["models_served"] = ["claude-sonnet-x"]
    assert model_call.finish(plan, returncode=0, stdout="", stderr="")["checks"]["only_requested_model_served"] is True

    cplan = model_call.prepare(["codex", "exec", "--model", "gpt-6-luna"], effort="medium", purpose="t")
    roll = tmp_path / "rollout.jsonl"
    roll.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(model_call, "_find_codex_thread",
                        lambda plan, ended: ({"id": "t-1", "rollout_path": str(roll)}, [], "found"))
    monkeypatch.setattr(L, "codex_rollout_file", lambda p: str(roll))
    monkeypatch.setattr(L, "extract_codex_db", lambda thread, turns: {"turns": [], "reasoning_items": 0})
    cfacts = {"effort_sent": "medium", "model_requested": "gpt-6-luna", "tools_called": [],
              "models_served": ["gpt-6-luna", "gpt-6-sol"]}
    monkeypatch.setattr(L, "extract_codex", lambda p: dict(cfacts))
    assert model_call.finish(cplan, returncode=0, stdout="", stderr="")["checks"]["only_requested_model_served"] is False
    cfacts["models_served"] = ["gpt-6-luna"]
    assert model_call.finish(cplan, returncode=0, stdout="", stderr="")["checks"]["only_requested_model_served"] is True


def test_a_held_out_block_strips_host_skills_text_from_checks_too():
    ci = {"host_skills_text": "SECRET", "n": 1}
    block = {"launch": {}, "facts": {"context_injections": dict(ci)}, "checks": {"context_injections": dict(ci)}}
    pub = model_call.public_block(block, held_out=True)
    assert "host_skills_text" not in pub["checks"]["context_injections"]
    assert "host_skills_text" not in pub["facts"]["context_injections"]
    assert block["checks"]["context_injections"]["host_skills_text"] == "SECRET"  # input not mutated
    assert model_call.public_block(block, held_out=False)["checks"]["context_injections"]["host_skills_text"] == "SECRET"
