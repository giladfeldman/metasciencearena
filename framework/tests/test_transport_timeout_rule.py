"""A transport timeout is our clock: never scored, never landed, and sized from the cap.

Owner's decision 2026-10-07 (d-545c62), on the 2026-10-06 recommendation:
  1. `report.apply_unscoreable_policy` excludes it instead of zeroing it
     (tested in test_unscoreable_policy.py and unscoreable-parity.test.ts);
  2. `scripts/qa_coverage_checks.py` C11 fails on any live timeout record;
  3. `framework run` / `retry-failed` default to a clock sized from the player's
     declared max_tokens.
This file covers 2 and 3, plus the one shared classifier all of them use.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from framework import timeouts
from framework.registry import load_registry

REPO = Path(__file__).resolve().parents[2]


def _qa():
    path = REPO / "scripts" / "qa_coverage_checks.py"
    if not path.is_file():
        # The public mirror ships this test but not the QA script (contract/public_manifest.txt).
        pytest.skip("scripts/qa_coverage_checks.py is not published - private checkout only")
    spec = importlib.util.spec_from_file_location("qa_coverage_checks_t", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rec(task: str, error: str | None) -> dict:
    bd = {"error": error} if error else {}
    return {"task_id": task, "player_id": "p", "score": {"primary": None if error else 0.5,
                                                          "breakdown": bd}}


def _arena(tmp_path: Path, recs: list[dict], sub: str = "runs/v1") -> Path:
    d = tmp_path / "arenas" / "fake-arena-v1" / sub
    d.mkdir(parents=True)
    (d / "p__revealed__x.jsonl").write_text(
        "\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
    return tmp_path / "arenas"


# ------------------------------------------------------------------ classifier
@pytest.mark.parametrize("msg", [
    "TimeoutExpired: Command '['claude']' timed out after 120 seconds",
    "ReadTimeout: HTTPConnectionPool(host='x', port=1): Read timed out. (read timeout=900)",
    "URLError: <urlopen error timed out>",
    "RuntimeError: subprocess exited 124",
    "RuntimeError: subprocess exited 137",
])
def test_every_known_timeout_wording_is_classified(msg):
    assert timeouts.is_transport_timeout(msg)


@pytest.mark.parametrize("msg", [
    "output_schema_violation: 'timeout' is not one of ['a']",
    "RuntimeError: 429 Too Many Requests",
    "ValueError: no parseable JSON",
    "",
    None,
])
def test_model_failures_and_other_transport_errors_are_not_timeouts(msg):
    assert not timeouts.is_transport_timeout(msg)


# ------------------------------------------------------------------ C11 gate
def test_c11_fails_on_a_single_live_timeout(tmp_path, monkeypatch):
    qa = _qa()
    monkeypatch.setattr(qa, "ARENAS", _arena(tmp_path, [
        _rec("t1", None), _rec("t2", "ReadTimeout: read timeout=900")]))
    fails = qa.check_transport_timeouts()
    assert len(fails) == 1 and "t2" in fails[0] and "retry-failed" in fails[0], fails


def test_c11_passes_a_clean_file_and_ignores_model_failures(tmp_path, monkeypatch):
    # Control: the same harness, no timeout, one schema violation quoting the word.
    qa = _qa()
    monkeypatch.setattr(qa, "ARENAS", _arena(tmp_path, [
        _rec("t1", None), _rec("t2", "output_schema_violation: 'timeout' is bad")]))
    assert qa.check_transport_timeouts() == []


def test_c11_skips_archived_evidence(tmp_path, monkeypatch):
    qa = _qa()
    monkeypatch.setattr(qa, "ARENAS", _arena(
        tmp_path, [_rec("t1", "ReadTimeout: read timeout=900")], sub="runs/v1/_archive"))
    assert qa.check_transport_timeouts() == []


def test_c11_is_registered():
    assert "C11" in _qa().CHECKS


# ------------------------------------------------------------------ the clock
def test_the_clock_is_sized_from_the_declared_cap():
    entry = {"player_id": "x", "openai_extra_body": {"max_tokens": 4096}}
    # 4096 / 6 tok/s + 180 s
    assert timeouts.timeout_for_entry(entry, fallback=120) == 862


def test_a_player_without_a_cap_keeps_the_fallback():
    assert timeouts.timeout_for_entry({"player_id": "x"}, fallback=120) == 120
    assert timeouts.timeout_for_entry({"player_id": "x"}, fallback=300) == 300


def test_the_clock_never_drops_below_the_fallback():
    entry = {"player_id": "x", "openai_extra_body": {"max_tokens": 16}}
    assert timeouts.timeout_for_entry(entry, fallback=300) == 300


def test_the_registry_declares_caps_where_this_module_reads_them():
    # Two-sided: if the registry moved its caps to another key, every clock would
    # silently fall back to 120 s. Measured 2026-10-07: 74 entries declare one.
    entries = load_registry(REPO / "players" / "registry.yaml")
    assert sum(1 for e in entries if timeouts.declared_max_tokens(e)) >= 50


def test_cli_defaults_leave_the_clock_to_the_registry():
    from framework import cli
    src = Path(cli.__file__).read_text(encoding="utf-8")
    assert 'p_run.add_argument("--timeout", type=int, default=None)' in src
    assert 'p_retry.add_argument("--timeout", type=int, default=None)' in src


def test_the_runner_plays_each_player_on_its_own_sized_clock(tmp_path, monkeypatch):
    """End to end through run_tournament, with the timeout the adapter actually received."""
    from framework import runner

    seen: dict[str, int] = {}
    real_build = runner.build_adapter

    def build(entry):
        # The stub's constructor takes no openai_extra_body; the RUNNER read the
        # cap from the entry before this point, which is what is under test.
        a = real_build({k: v for k, v in entry.items() if k != "openai_extra_body"})
        play = a.play_task

        def recording(envelope, *, timeout_s):
            seen[entry["player_id"]] = timeout_s
            return play(envelope, timeout_s=timeout_s)

        a.play_task = recording
        return a

    monkeypatch.setattr(runner, "build_adapter", build)
    fake = REPO / "framework" / "tests" / "fixtures" / "fake_arena"
    if not fake.is_dir():
        pytest.skip("fake_arena fixture not present")
    reg = tmp_path / "registry.yaml"
    reg.write_text(
        "- {player_id: capped, player_type: tool, player_version: '1', adapter_class: StubPassAdapter,"
        " confidence_strategy: implicit-1.0, deterministic: true, openai_extra_body: {max_tokens: 4096}}\n"
        "- {player_id: plain, player_type: tool, player_version: '1', adapter_class: StubPassAdapter,"
        " confidence_strategy: implicit-1.0, deterministic: true}\n", encoding="utf-8")
    runner.run_tournament(arena_dir=fake, task_set_version="v1", registry_path=reg,
                          player_ids=["capped", "plain"], output_path=tmp_path / "o.jsonl",
                          trials=1, timeout_s=None, max_tasks=1)
    assert seen == {"capped": 862, "plain": 120}


# ------------------------------------------------------------------ consult 2026-10-07
# Two findings from the Sonnet consult on this change, both reproduced before fixing.

def test_model_text_quoted_after_the_first_segment_is_not_our_clock():
    # Finding 2: the old substring match excused this as a timeout.
    msg = "ValueError: no parseable JSON in CLI output (len=40): 'the request timed out so'"
    assert not timeouts.is_transport_timeout(msg)
    assert not timeouts.is_transport_timeout(
        "RuntimeError: score.R produced invalid JSON: x; raw: exited 124 timeout")


def test_a_windows_command_path_does_not_hide_the_exit_code():
    assert timeouts.is_transport_timeout(
        "RuntimeError: C:/Users/x/.local/bin/claude.EXE exited 124: killed")
    assert timeouts.is_transport_timeout(
        r"RuntimeError: C:\Users\x\claude.EXE exited 137: ")


def test_a_held_out_exit_124_keeps_its_timeout_class():
    # Finding 1: sanitize_error cut it to "RuntimeError", which the policy zeroed.
    from framework.holdout import sanitize_error
    out = sanitize_error("RuntimeError: claude exited 124: <stderr with task text>")
    assert out == "TransportTimeout" and timeouts.is_transport_timeout(out)
    # Controls: a typed timeout keeps its own name; a real crash stays a crash.
    assert sanitize_error("TimeoutExpired: Command 'x' timed out after 9 seconds") == "TimeoutExpired"
    assert sanitize_error("RuntimeError: claude exited 1: boom") == "RuntimeError"


def test_the_bare_word_timeout_in_free_text_is_not_our_clock():
    # sciencearena-review 2026-10-08, reproduced: True before this fix.
    assert not timeouts.is_transport_timeout("ValueError: invalid timeout value in config")
    assert timeouts.is_transport_timeout("RuntimeError: timeout exited 124: \n> build")


def test_clock_is_sized_from_a_declared_claude_cli_ceiling():
    # d-60b28c: without this the clock stayed at the 120 s default and every long
    # publication-bias v2 answer timed out (measured 2026-10-08).
    from framework import timeouts
    entry = {"cli_max_output_tokens": 200000}
    assert timeouts.declared_max_tokens(entry) == 200000
    assert timeouts.timeout_for_entry(entry, 120) == timeouts.timeout_for_cap(200000) > 120
    assert timeouts.declared_max_tokens({}) is None
    assert timeouts.declared_max_tokens({"cli_max_output_tokens": True}) is None
