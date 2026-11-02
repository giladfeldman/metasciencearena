"""The CLI session-log reader and the past-run recovery that uses it.

Synthetic logs only: the shapes below are copied from real Claude Code 2.1.310 transcripts and
Codex 0.164.0 rollouts / SQLite stores (field names, nesting), with invented values.
"""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
import sys
from pathlib import Path

import pytest

from framework import cli_session_log as L

REPO = Path(__file__).resolve().parents[2]


def _jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


CLAUDE_ROWS = [
    {"type": "attachment", "timestamp": "2026-10-30T10:00:00.000Z", "version": "2.1.310",
     "entrypoint": "claude-desktop", "sessionId": "s-1",
     "attachment": {"context": {"userEmail": "someone@example.org"}}},
    {"type": "api-request", "timestamp": "2026-10-30T10:00:01.000Z",
     "params": {"model": "claude-haiku-5-5", "max_tokens": 128000,
                "thinking": {"type": "adaptive", "display": "omitted"},
                "output_config": {"effort": "medium"}, "betas": ["b1", "b2"]}},
    {"type": "assistant", "timestamp": "2026-10-30T10:00:09.000Z", "requestId": "req_1", "effort": "medium",
     "message": {"id": "msg_1", "model": "claude-haiku-5-5", "stop_reason": "end_turn",
                 "content": [{"type": "text", "text": "the answer"}],
                 "usage": {"input_tokens": 10, "output_tokens": 20, "cache_read_input_tokens": 5,
                           "service_tier": "standard"}}},
]


def test_claude_transcript_effort_and_settings(tmp_path):
    f = L.extract_claude(_jsonl(tmp_path / "s-1.jsonl", CLAUDE_ROWS))
    assert f["effort_sent"] == "medium"
    assert f["effort_reported"] == "medium"
    assert f["thinking_type"] == "adaptive"
    assert f["max_tokens_sent"] == 128000
    assert f["harness_version"] == "2.1.310"
    assert f["context_injections"]["user_email"] is True


def test_claude_never_returns_the_email_or_any_text(tmp_path):
    f = L.extract_claude(_jsonl(tmp_path / "s-1.jsonl", CLAUDE_ROWS))
    blob = json.dumps(f)
    assert "someone@example.org" not in blob
    assert "the answer" not in blob


def test_claude_older_cli_without_request_log_is_not_in_log(tmp_path):
    rows = [r for r in CLAUDE_ROWS if r["type"] != "api-request"]
    f = L.extract_claude(_jsonl(tmp_path / "s-1.jsonl", rows))
    assert f["effort_sent"] == L.NOT_IN_LOG          # not "absent": the request was not logged at all
    assert f["effort_reported"] == "medium"


CODEX_ROWS = [
    {"type": "session_meta", "timestamp": "2026-10-30T10:00:00.000Z",
     "payload": {"id": "t-1", "cli_version": "0.164.0", "originator": "codex_exec", "model_provider": "openai",
                 "cwd": "C:/x/sciencearena-hermetic/player", "base_instructions": {"text": "be helpful"}}},
    {"type": "turn_context", "timestamp": "2026-10-30T10:00:01.000Z",
     "payload": {"model": "gpt-6-luna", "summary": "auto", "approval_policy": "never",
                 "sandbox_policy": {"type": "read-only"},
                 "collaboration_mode": {"settings": {"reasoning_effort": None}}}},
    {"type": "event_msg", "timestamp": "2026-10-30T10:00:08.000Z",
     "payload": {"type": "token_count", "rate_limits": {"plan_type": "plus"},
                 "info": {"total_token_usage": {"input_tokens": 100, "output_tokens": 7,
                                                "reasoning_output_tokens": 0}}}},
    {"type": "event_msg", "timestamp": "2026-10-30T10:00:09.000Z",
     "payload": {"type": "task_complete", "duration_ms": 9000, "last_agent_message": "final words"}},
]


def test_codex_rollout_absent_effort_is_typed(tmp_path):
    f = L.extract_codex(_jsonl(tmp_path / "rollout-t-1.jsonl", CODEX_ROWS))
    assert f["effort_sent"] == L.ABSENT
    assert f["model_requested"] == "gpt-6-luna"
    assert f["usage"]["reasoning_tokens"] == 0
    assert f["timing"]["turn_ms"] == 9000
    assert "final words" not in json.dumps(f)


def test_codex_compressed_rollout_reads_like_the_plain_one(tmp_path):
    from compression import zstd
    plain = _jsonl(tmp_path / "rollout-t-1.jsonl", CODEX_ROWS)
    z = tmp_path / "rollout-t-2.jsonl.zst"
    z.write_bytes(zstd.compress(plain.read_bytes()))
    a, b = L.extract_codex(plain), L.extract_codex(z)
    for k in ("effort_sent", "model_requested", "usage", "timing", "harness_version"):
        assert a[k] == b[k]
    assert L.final_codex_message(z) == "final words"


def test_rollout_file_found_when_codex_compressed_it(tmp_path):
    recorded = tmp_path / "sessions" / "rollout-t-1.jsonl"
    recorded.parent.mkdir()
    (tmp_path / "sessions" / "rollout-t-1.jsonl.zst").write_bytes(b"x")
    assert L.codex_rollout_file(str(recorded)).endswith(".jsonl.zst")
    assert L.codex_rollout_file(str(tmp_path / "sessions" / "nope.jsonl")) is None


def _codex_stores(home: Path) -> None:
    s = sqlite3.connect(home / L.CODEX_STATE_DB)
    s.execute("create table threads (id text, rollout_path text, created_at_ms int, updated_at_ms int, "
              "source text, model_provider text, cwd text, sandbox_policy text, approval_mode text, "
              "tokens_used int, cli_version text, first_user_message text, model text, reasoning_effort text)")
    s.execute("insert into threads values ('t-1', 'gone.jsonl', 1793000000000, 1793000009000, 'exec', 'openai', "
              "'C:/x/sciencearena-hermetic/player', ?, 'never', 107, '0.164.0', 'task text', 'gpt-6.1-sol', 'high')",
              (json.dumps({"type": "managed", "network": "restricted",
                           "file_system": {"type": "restricted", "entries": [{"access": "read"}]}}),))
    s.execute("insert into threads values ('t-2', 'x', 1, 2, 'exec', 'openai', 'C:/elsewhere', '{}', 'never', "
              "1, '0.164.0', '', 'gpt-6.1-sol', 'high')")
    s.commit()
    h = sqlite3.connect(home / L.CODEX_HISTORY_DB)
    h.execute("create table thread_turns (thread_id text, turn_id text, rollout_ordinal int, status text, "
              "error_json text, started_at int, completed_at int, duration_ms int)")
    h.execute("create table thread_items (thread_id text, turn_id text, item_id text, rollout_ordinal int, "
              "item_type text, item_json text)")
    h.execute("insert into thread_turns values ('t-1', 'u-1', 0, 'failed', ?, 1, 2, 9000)",
              (json.dumps({"codexErrorInfo": "usageLimitExceeded"}),))
    h.execute("insert into thread_items values ('t-1', 'u-1', 'i1', 0, 'reasoning', '{}')")
    h.execute("insert into thread_items values ('t-1', 'u-1', 'i2', 1, 'agentMessage', ?)",
              (json.dumps({"text": "the reply"}),))
    h.commit()


def test_codex_store_facts_when_the_rollout_is_gone(tmp_path):
    _codex_stores(tmp_path)
    threads = L.codex_threads(tmp_path)
    assert [t["id"] for t in threads] == ["t-1"]          # only arena folders
    turns = L.codex_thread_turns(["t-1"], tmp_path)
    f = L.extract_codex_db(threads[0], turns["t-1"])
    assert f["effort_sent"] == "high"
    assert f["source"] == "codex-state-db"
    assert f["turns"] == [{"status": "failed", "error": "usageLimitExceeded", "duration_ms": 9000}]
    assert f["sandbox"] == "managed/restricted/read"
    assert f["reasoning_items"] == 1
    assert L.final_codex_message_db(turns["t-1"]) == "the reply"
    assert "task text" not in json.dumps(f) and "the reply" not in json.dumps(f)


def test_codex_store_is_opened_read_only(tmp_path):
    _codex_stores(tmp_path)
    con = L.codex_db(L.CODEX_STATE_DB, tmp_path)
    with pytest.raises(sqlite3.OperationalError):
        con.execute("delete from threads")


def test_public_view_strips_call_specific_facts_for_held_out(tmp_path):
    f = L.extract_codex(_jsonl(tmp_path / "rollout-t-1.jsonl", CODEX_ROWS))
    pub = L.public_view(f, held_out=True)
    assert "usage" not in pub and "timing" not in pub and "session_id" not in pub
    assert pub["effort_sent"] == L.ABSENT
    assert L.public_view(f, held_out=False)["usage"] == f["usage"]


def test_public_view_handles_an_aggregate():
    agg = {"context_injections": [{"host_skills": True, "host_skills_text": {"sha256": "a"}}], "usage": {"x": 1}}
    pub = L.public_view(agg, held_out=True)
    assert pub["context_injections"] == [{"host_skills": True}]


# ------------------------------------------------------------------ linking

@pytest.fixture(scope="module")
def R():
    # scripts/ is not part of the public mirror (mirrored-suite run, 2026-11-02): the tests that
    # use this fixture skip there by name instead of erroring.
    if not (REPO / "scripts" / "recover_cli_provenance.py").is_file():
        pytest.skip("scripts/recover_cli_provenance.py is not in the public mirror -- NOT verified here")
    sys.path.insert(0, str(REPO))
    from scripts import recover_cli_provenance as mod
    return mod


def _t(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s).replace(tzinfo=dt.timezone.utc)


def test_nearest_end_refuses_a_close_runner_up(R):
    end = _t("2026-10-30T10:00:00")
    a = {"ended_dt": end - dt.timedelta(seconds=1)}
    b = {"ended_dt": end - dt.timedelta(seconds=2)}
    assert R._nearest_end([a, b], end) is None              # 1 s apart < END_MARGIN_S
    c = {"ended_dt": end - dt.timedelta(seconds=10)}
    assert R._nearest_end([a, c], end) is a


def test_regcheck_calls_after_the_record_belong_to_the_next_task(R):
    rec = {"timestamp_utc": "2026-10-30T10:05:00Z", "latency_ms": 60_000}
    idx = [{"regcheck": True, "started_dt": _t("2026-10-30T10:04:10")},
           {"regcheck": True, "started_dt": _t("2026-10-30T10:05:02")},   # next task
           {"regcheck": False, "started_dt": _t("2026-10-30T10:04:20")}]
    assert R.link_regcheck(rec, idx) == [idx[0]]


def test_candidates_agree_keeps_settings_only(R, tmp_path):
    a = _jsonl(tmp_path / "a.jsonl", CLAUDE_ROWS)
    b = _jsonl(tmp_path / "b.jsonl", CLAUDE_ROWS)
    f = R.settings_if_candidates_agree([str(a), str(b)])
    assert f["effort_sent"] == "medium" and f["candidates"] == 2
    assert "usage" not in f and "request_ids" not in f
    other = [dict(r) for r in CLAUDE_ROWS]
    other[1] = {**other[1], "params": {**other[1]["params"], "output_config": {"effort": "high"}}}
    c = _jsonl(tmp_path / "c.jsonl", other)
    assert R.settings_if_candidates_agree([str(a), str(c)]) is None


def test_purpose_masks_ids(R):
    assert R.purpose_of("C:/T/sciencearena-hermetic/goldrun-adj-sol-1a2b3c4d") == "goldrun-adj-sol-#"
    assert R.purpose_of("C:/Users/x/ScienceArena").startswith("repository-folder")


def test_codex_records_the_served_model_and_any_reroute(tmp_path):
    """Go review 2026-11-01 (Sol): Codex can reroute a turn to another model (its ModelReroute event:
    from_model, to_model, reason). The reader ignored it, so a substituted answer read as the requested
    model's."""
    reroute = {"type": "event_msg", "timestamp": "2026-10-30T10:00:05.000Z",
               "payload": {"type": "model_reroute", "from_model": "gpt-6-luna", "to_model": "gpt-6-sol",
                           "reason": "load"}}
    f = L.extract_codex(_jsonl(tmp_path / "rollout-t-1.jsonl", CODEX_ROWS[:2] + [reroute] + CODEX_ROWS[2:]))
    assert f["models_served"] == ["gpt-6-luna", "gpt-6-sol"]
    assert f["model_reroutes"] == [{"from": "gpt-6-luna", "to": "gpt-6-sol", "reason": "load"}]
    plain = L.extract_codex(_jsonl(tmp_path / "rollout-t-2.jsonl", CODEX_ROWS))
    assert plain["models_served"] == ["gpt-6-luna"] and plain["model_reroutes"] == []


def test_codex_reported_effort_is_not_a_copy_of_the_sent_one(tmp_path):
    """Go review 2026-11-01 (both seats): Codex logs the setting once (turn_context); the reader copied it
    into effort_reported, which then read as a second, independent acknowledgement."""
    rows = [dict(r) for r in CODEX_ROWS]
    rows[1] = {**rows[1], "payload": {**rows[1]["payload"], "effort": "medium"}}
    f = L.extract_codex(_jsonl(tmp_path / "rollout-t-1.jsonl", rows))
    assert f["effort_sent"] == "medium"
    assert f["effort_reported"] == L.NOT_IN_LOG
