"""Read what a subscription CLI itself logged about one model call.

Claude Code and Codex each write a session log for every call (Claude Code:
``~/.claude/projects/<cwd-key>/<session_id>.jsonl``; Codex:
``$CODEX_HOME/sessions/YYYY/MM/DD/rollout-*.jsonl``). Those logs are the only
place that records the request the harness actually sent: the effort level,
the thinking mode, the output ceiling, the beta features, the system prompt the
harness added, the tools it offered, the sandbox, plus the provider's request
and response ids, token counts including reasoning tokens, and timings. The
CLI's stdout envelope carries only a fraction of this, and Claude Code does
not echo the effort at all (measured 2026-11-01, Claude Code 2.1.316).

One reader serves two callers, so the recovered past and the recorded future
can never disagree: ``scripts/recover_cli_provenance.py`` (past records) and
the CLI adapters (every new call).

Everything returned is DERIVED: numbers, ids, labels, booleans and sha256
digests. No prompt, answer, system-prompt or task text is ever returned; a text
is represented by ``{"sha256", "chars"}``. Values injected from the launching
environment (the owner's e-mail, a session's scratch folder, an organisation
id) are reported as booleans only, never their values.

Absence is typed (``NOT_IN_LOG``): an older CLI version that did not log a field
is not the same fact as a request that did not set it (``ABSENT``).
"""
from __future__ import annotations

import datetime as _dt
import glob
import hashlib
import json
import os
from pathlib import Path

#: 2 (2026-11-01): Codex facts gain models_served and model_reroutes, and its effort_reported is
#: NOT_IN_LOG instead of a copy of effort_sent (go review, docs/consults/2026-11-01-power-review-go).
FACTS_SCHEMA_VERSION = 2
#: The log has no entry that could carry this field (older CLI version, or a log type without it).
NOT_IN_LOG = "not-in-log"
#: The log shows the request and the request did not carry this setting.
ABSENT = "absent"


def _sha(text: str) -> dict:
    return {"sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "chars": len(text)}


def file_sha256(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _open_text(path: str | os.PathLike):
    """Open a log as text; Codex compresses older rollouts to ``.jsonl.zst`` (zstandard)."""
    if str(path).endswith(".zst"):
        import io
        from compression import zstd  # standard library from Python 3.14
        return io.TextIOWrapper(zstd.open(path, "rb"), encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")


def _lines(path: str | os.PathLike):
    with _open_text(path) as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


# ---------------------------------------------------------------- Claude Code

def claude_projects_root() -> Path:
    base = os.environ.get("CLAUDE_CONFIG_DIR") or os.path.join(os.path.expanduser("~"), ".claude")
    return Path(base) / "projects"


def find_claude_transcript(session_id: str, root: Path | None = None) -> Path | None:
    """The transcript for a Claude Code session id, in any project folder."""
    if not session_id or any(c in session_id for c in "/\\*?"):
        return None
    hits = glob.glob(str((root or claude_projects_root()) / "*" / f"{session_id}.jsonl"))
    return Path(hits[0]) if len(hits) == 1 else None


def extract_claude(path: str | os.PathLike) -> dict:
    """Facts from one Claude Code transcript (see module docstring)."""
    f: dict = {
        "facts_schema_version": FACTS_SCHEMA_VERSION,
        "source": "claude-code-transcript",
        "source_sha256": file_sha256(path),
        "harness": "claude-code",
        "harness_version": NOT_IN_LOG,
        "entrypoint": NOT_IN_LOG,
        "session_id": NOT_IN_LOG,
        "model_requested": NOT_IN_LOG,
        "models_served": [],
        "models_billed": [],
        "advisor_model_configured": NOT_IN_LOG,
        "effort_sent": NOT_IN_LOG,
        "effort_reported": NOT_IN_LOG,
        "thinking_type": NOT_IN_LOG,
        "thinking_display": NOT_IN_LOG,
        "max_tokens_sent": NOT_IN_LOG,
        "temperature_sent": NOT_IN_LOG,
        "top_p_sent": NOT_IN_LOG,
        "top_k_sent": NOT_IN_LOG,
        "betas": NOT_IN_LOG,
        "context_management": NOT_IN_LOG,
        "cache_ttl": NOT_IN_LOG,
        "system_prompt": NOT_IN_LOG,
        "tools_offered": NOT_IN_LOG,
        "tools_called": [],
        "request_ids": [],
        "message_ids": [],
        "request_digests": [],
        "stop_reasons": [],
        "service_tier": NOT_IN_LOG,
        "inference_geo": NOT_IN_LOG,
        "speed": NOT_IN_LOG,
        "usage": NOT_IN_LOG,
        "timing": {},
        "cost_usd_list_equivalent": NOT_IN_LOG,
        "context_injections": {"user_email": False, "scratchpad_directory": False,
                               "organization_uuid": False, "environment_snapshot": False,
                               "model_identity": False},
        "api_requests": 0,
        "synthetic_messages": 0,
        "first_timestamp": None,
        "last_timestamp": None,
    }
    usage_by_msg: dict[str, dict] = {}
    thinking_ms = 0
    betas: set[str] = set()
    efforts_sent: list = []
    efforts_rep: list = []
    for o in _lines(path):
        t = o.get("type")
        ts = o.get("timestamp")
        if isinstance(ts, str):
            f["first_timestamp"] = ts if not f["first_timestamp"] or ts < f["first_timestamp"] else f["first_timestamp"]
            f["last_timestamp"] = ts if not f["last_timestamp"] or ts > f["last_timestamp"] else f["last_timestamp"]
        if o.get("version") and f["harness_version"] == NOT_IN_LOG:
            f["harness_version"] = o["version"]
        if o.get("entrypoint") and f["entrypoint"] == NOT_IN_LOG:
            f["entrypoint"] = o["entrypoint"]
        if o.get("sessionId") and f["session_id"] == NOT_IN_LOG:
            f["session_id"] = o["sessionId"]
        if t == "api-request":
            f["api_requests"] += 1
            p = o.get("params") or {}
            f["model_requested"] = p.get("model", f["model_requested"])
            f["max_tokens_sent"] = p.get("max_tokens", ABSENT)
            f["temperature_sent"] = p.get("temperature", ABSENT)
            f["top_p_sent"] = p.get("top_p", ABSENT)
            f["top_k_sent"] = p.get("top_k", ABSENT)
            th = p.get("thinking")
            if isinstance(th, dict):
                f["thinking_type"] = th.get("type", ABSENT)
                f["thinking_display"] = th.get("display", ABSENT)
                if "budget_tokens" in th:
                    f["thinking_budget_sent"] = th["budget_tokens"]
            else:
                f["thinking_type"] = ABSENT
            efforts_sent.append((p.get("output_config") or {}).get("effort", ABSENT))
            betas.update(str(b) for b in (p.get("betas") or []))
            cm = p.get("context_management") or {}
            f["context_management"] = sorted({str(e.get("type")) for e in cm.get("edits") or []}) or ABSENT
            ttl = sorted({str((c.get("value") or {}).get("ttl")) for c in o.get("cacheControl") or []})
            f["cache_ttl"] = ttl or ABSENT
            if o.get("requestDigest"):
                f["request_digests"].append(o["requestDigest"])
        elif t == "api-request-shape":
            shape = o.get("shape") or {}
            parts = [str(s.get("text", "")) for s in shape.get("system") or [] if isinstance(s, dict)]
            sp = _sha("\n".join(parts))
            sp["parts"] = len(parts)
            sp["shape_hash"] = o.get("shapeHash")
            f["system_prompt"] = sp
            tools = shape.get("tools") or []
            f["tools_offered"] = sorted(str(x.get("name")) for x in tools if isinstance(x, dict))
        elif t == "assistant":
            m = o.get("message") or {}
            model = m.get("model")
            if model == "<synthetic>":
                f["synthetic_messages"] += 1
            elif model and model not in f["models_served"]:
                f["models_served"].append(model)
            if o.get("requestedModel") and f["model_requested"] == NOT_IN_LOG:
                f["model_requested"] = o["requestedModel"]
            if o.get("advisorModel"):
                f["advisor_model_configured"] = o["advisorModel"]
            rep = o.get("perTurnEffort") or o.get("effort")
            if rep:
                efforts_rep.append(rep)
            if o.get("requestId") and o["requestId"] not in f["request_ids"]:
                f["request_ids"].append(o["requestId"])
            mid = m.get("id")
            if mid and mid not in f["message_ids"]:
                f["message_ids"].append(mid)
            if m.get("stop_reason") and (mid not in usage_by_msg):
                f["stop_reasons"].append(m["stop_reason"])
            for b in m.get("content") or []:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    f["tools_called"].append(str(b.get("name")))
            u = m.get("usage")
            if mid and isinstance(u, dict) and mid not in usage_by_msg:
                usage_by_msg[mid] = u
                if isinstance(o.get("thinkingDurationMs"), (int, float)):
                    thinking_ms += o["thinkingDurationMs"]
        elif t == "attachment":
            a = o.get("attachment") or {}
            ctx = a.get("context") or {}
            if ctx.get("userEmail"):
                f["context_injections"]["user_email"] = True
            snap = a.get("snapshot") or {}
            if snap:
                f["context_injections"]["environment_snapshot"] = True
                if snap.get("scratchpadDirectory"):
                    f["context_injections"]["scratchpad_directory"] = True
            if a.get("organizationUuid"):
                f["context_injections"]["organization_uuid"] = True
            if a.get("identity"):
                f["context_injections"]["model_identity"] = True
        elif t == "cost-state":
            f["cost_usd_list_equivalent"] = o.get("totalCostUSD", f["cost_usd_list_equivalent"])
            f["models_billed"] = sorted((o.get("modelUsage") or {}).keys())
            api = o.get("totalAPIDuration")
            api_nr = o.get("totalAPIDurationWithoutRetries")
            f["timing"].update({"api_ms": api, "api_ms_without_retries": api_nr,
                                "retry_ms": (api - api_nr) if isinstance(api, (int, float)) and isinstance(api_nr, (int, float)) else NOT_IN_LOG,
                                "session_ms": o.get("totalDuration")})
    if usage_by_msg:
        tot = {"input_tokens": 0, "cache_read_tokens": 0, "cache_creation_1h_tokens": 0,
               "cache_creation_5m_tokens": 0, "output_tokens": 0, "thinking_tokens": 0}
        tiers, geos, speeds = set(), set(), set()
        thinking_known = False
        for u in usage_by_msg.values():
            tot["input_tokens"] += u.get("input_tokens") or 0
            tot["cache_read_tokens"] += u.get("cache_read_input_tokens") or 0
            cc = u.get("cache_creation") or {}
            tot["cache_creation_1h_tokens"] += cc.get("ephemeral_1h_input_tokens") or 0
            tot["cache_creation_5m_tokens"] += cc.get("ephemeral_5m_input_tokens") or 0
            if not cc and u.get("cache_creation_input_tokens"):
                tot["cache_creation_5m_tokens"] += u["cache_creation_input_tokens"]
            tot["output_tokens"] += u.get("output_tokens") or 0
            det = u.get("output_tokens_details") or {}
            if "thinking_tokens" in det:
                thinking_known = True
                tot["thinking_tokens"] += det.get("thinking_tokens") or 0
            tiers.add(u.get("service_tier"))
            geos.add(u.get("inference_geo"))
            speeds.add(u.get("speed"))
        if not thinking_known:
            tot["thinking_tokens"] = NOT_IN_LOG
        f["usage"] = tot
        f["service_tier"] = ",".join(sorted(str(x) for x in tiers if x)) or NOT_IN_LOG
        f["inference_geo"] = ",".join(sorted(str(x) for x in geos if x)) or NOT_IN_LOG
        f["speed"] = ",".join(sorted(str(x) for x in speeds if x)) or NOT_IN_LOG
        f["timing"]["thinking_ms"] = thinking_ms
    if betas:
        f["betas"] = sorted(betas)
    elif f["api_requests"]:
        f["betas"] = ABSENT
    if efforts_sent:
        f["effort_sent"] = efforts_sent[0] if len(set(efforts_sent)) == 1 else efforts_sent
    if efforts_rep:
        f["effort_reported"] = efforts_rep[0] if len(set(efforts_rep)) == 1 else sorted(set(efforts_rep))
    return f


# ---------------------------------------------------------------- Codex

def codex_home() -> Path:
    return Path(os.environ.get("CODEX_HOME") or os.path.join(os.path.expanduser("~"), ".codex"))


def find_codex_rollout(thread_id: str, home: Path | None = None) -> Path | None:
    if not thread_id or any(c in thread_id for c in "/\\*?"):
        return None
    root = home or codex_home()
    hits = []
    for sub in ("sessions", "archived_sessions"):
        for ext in ("jsonl", "jsonl.zst"):
            hits += glob.glob(str(root / sub / "**" / f"rollout-*{thread_id}.{ext}"), recursive=True)
    return Path(hits[0]) if len(hits) == 1 else None


def codex_rollout_file(recorded_path: str) -> str | None:
    """Where a thread's rollout is now: as recorded, compressed beside it (``.zst``),
    or moved to ``archived_sessions``."""
    sep = os.sep
    archived = recorded_path.replace(f"{sep}sessions{sep}", f"{sep}archived_sessions{sep}")
    for cand in (recorded_path, recorded_path + ".zst", archived, archived + ".zst"):
        if cand and os.path.exists(cand):
            return cand
    return None


def extract_codex(path: str | os.PathLike) -> dict:
    """Facts from one Codex rollout (see module docstring)."""
    f: dict = {
        "facts_schema_version": FACTS_SCHEMA_VERSION,
        "source": "codex-rollout",
        "source_sha256": file_sha256(path),
        "harness": "codex-cli",
        "harness_version": NOT_IN_LOG,
        "entrypoint": NOT_IN_LOG,
        "session_id": NOT_IN_LOG,
        "model_requested": NOT_IN_LOG,
        # Every model a turn ran on: turn_context's model, plus the target of any ModelReroute event
        # (go review 2026-11-01: a rerouted answer read as the requested model's).
        "models_served": [],
        "model_reroutes": [],
        "model_provider": NOT_IN_LOG,
        "effort_sent": NOT_IN_LOG,
        "effort_reported": NOT_IN_LOG,
        "reasoning_summary": NOT_IN_LOG,
        "base_instructions": NOT_IN_LOG,
        "developer_instructions": NOT_IN_LOG,
        "model_instructions_hash": NOT_IN_LOG,
        "model_catalog_hash": NOT_IN_LOG,
        "tools_offered": NOT_IN_LOG,
        "tools_called": [],
        "sandbox": NOT_IN_LOG,
        "network": NOT_IN_LOG,
        "approval_policy": NOT_IN_LOG,
        "model_context_window": NOT_IN_LOG,
        "response_ids": [],
        "usage": NOT_IN_LOG,
        "timing": {},
        "rate_limit_plan": NOT_IN_LOG,
        "context_injections": {"host_skills": False, "host_skills_text": None,
                               "timezone": None, "shell": None},
        "first_timestamp": None,
        "last_timestamp": None,
    }
    dev_texts: list[str] = []
    for o in _lines(path):
        t = o.get("type")
        p = o.get("payload") if isinstance(o.get("payload"), dict) else {}
        pt = p.get("type")
        ts = o.get("timestamp")
        if isinstance(ts, str):
            f["first_timestamp"] = ts if not f["first_timestamp"] or ts < f["first_timestamp"] else f["first_timestamp"]
            f["last_timestamp"] = ts if not f["last_timestamp"] or ts > f["last_timestamp"] else f["last_timestamp"]
        if t == "session_meta":
            f["harness_version"] = p.get("cli_version", NOT_IN_LOG)
            f["entrypoint"] = p.get("originator", NOT_IN_LOG)
            f["session_id"] = p.get("session_id") or p.get("id") or NOT_IN_LOG
            f["model_provider"] = p.get("model_provider", NOT_IN_LOG)
            bi = (p.get("base_instructions") or {})
            if isinstance(bi.get("text"), str):
                f["base_instructions"] = _sha(bi["text"])
        elif t == "event_msg" and pt == "task_started":
            f["model_context_window"] = p.get("model_context_window", NOT_IN_LOG)
        elif t == "response_item" and pt == "additional_tools":
            names = []
            for ns in p.get("tools") or []:
                for tt in ns.get("tools") or []:
                    if tt.get("name"):
                        names.append(f"{ns.get('name')}.{tt['name']}")
            f["tools_offered"] = sorted(names)
        elif t == "response_item" and pt == "message" and p.get("role") == "developer":
            for c in p.get("content") or []:
                if isinstance(c, dict) and isinstance(c.get("text"), str):
                    dev_texts.append(c["text"])
        elif t == "response_item" and pt in ("function_call", "custom_tool_call", "local_shell_call",
                                             "web_search_call", "tool_search_call"):
            f["tools_called"].append(str(p.get("name") or pt))
        elif t == "world_state":
            st = p.get("state") or {}
            hs = (st.get("host_skills") or {}).get("body")
            if isinstance(hs, str) and hs:
                f["context_injections"]["host_skills"] = True
                f["context_injections"]["host_skills_text"] = _sha(hs)
            env = (st.get("environments") or {})
            f["context_injections"]["timezone"] = env.get("timezone")
            loc = ((env.get("environments") or {}).get("local") or {})
            f["context_injections"]["shell"] = loc.get("shell")
            # Older Codex versions log `model` as a bare string here.
            wm = st.get("model")
            f["model_instructions_hash"] = wm.get("instructions", NOT_IN_LOG) if isinstance(wm, dict) else NOT_IN_LOG
            f["model_catalog_hash"] = st.get("model_catalog", NOT_IN_LOG)
        elif t == "event_msg" and pt == "model_reroute":
            f["model_reroutes"].append({"from": p.get("from_model"), "to": p.get("to_model"),
                                        "reason": p.get("reason")})
            if p.get("to_model") and p["to_model"] not in f["models_served"]:
                f["models_served"].append(p["to_model"])
        elif t == "turn_context":
            f["model_requested"] = p.get("model", f["model_requested"])
            if p.get("model") and p["model"] not in f["models_served"]:
                f["models_served"].append(p["model"])
            settings = ((p.get("collaboration_mode") or {}).get("settings") or {})
            if "effort" in p:
                f["effort_sent"] = p["effort"]
            elif "reasoning_effort" in settings:
                f["effort_sent"] = settings["reasoning_effort"] if settings["reasoning_effort"] is not None else ABSENT
            else:
                f["effort_sent"] = ABSENT
            # Codex logs the setting once, here; there is no separate acknowledgement to read, so
            # effort_reported stays NOT_IN_LOG rather than repeating effort_sent (go review 2026-11-01).
            f["reasoning_summary"] = p.get("summary", ABSENT)
            f["sandbox"] = (p.get("sandbox_policy") or {}).get("type", NOT_IN_LOG)
            f["network"] = (p.get("permission_profile") or {}).get("network", NOT_IN_LOG)
            f["approval_policy"] = p.get("approval_policy", NOT_IN_LOG)
        elif t == "token_usage_record":
            if p.get("response_id") and p["response_id"] not in f["response_ids"]:
                f["response_ids"].append(p["response_id"])
            u = p.get("thread_token_usage") or p.get("usage") or {}
            f["usage"] = {"input_tokens": u.get("input_tokens"),
                          "cache_read_tokens": u.get("cached_input_tokens"),
                          "cache_write_tokens": u.get("cache_write_input_tokens"),
                          "output_tokens": u.get("output_tokens"),
                          "reasoning_tokens": u.get("reasoning_output_tokens", NOT_IN_LOG)}
        elif t == "event_msg" and pt == "token_count":
            rl = p.get("rate_limits") or {}
            if rl.get("plan_type"):
                f["rate_limit_plan"] = rl["plan_type"]
            if f["usage"] == NOT_IN_LOG:
                u = ((p.get("info") or {}).get("total_token_usage") or {})
                if u:
                    f["usage"] = {"input_tokens": u.get("input_tokens"),
                                  "cache_read_tokens": u.get("cached_input_tokens"),
                                  "cache_write_tokens": u.get("cache_write_input_tokens"),
                                  "output_tokens": u.get("output_tokens"),
                                  "reasoning_tokens": u.get("reasoning_output_tokens", NOT_IN_LOG)}
        elif t == "event_msg" and pt == "task_complete":
            f["timing"].update({"turn_ms": p.get("duration_ms"), "ttft_ms": p.get("time_to_first_token_ms")})
    if dev_texts:
        f["developer_instructions"] = _sha("\n".join(dev_texts))
    return f


# Codex also keeps two SQLite stores that outlive deleted rollouts (measured
# 2026-11-01, Codex 0.166.0): ``state_5.sqlite`` (one ``threads`` row per
# session: model, reasoning_effort, cli_version, tokens_used, sandbox, approval,
# times) and ``thread_history_1.sqlite`` (``thread_turns``: status, error,
# duration; ``thread_items``: messages, reasoning, commands). 651 hermetic
# sessions of 2,550 had no plain ``.jsonl`` rollout any more: Codex had compressed
# them to ``.jsonl.zst``. Every arena session since May survives in one form.

CODEX_STATE_DB = "state_5.sqlite"
CODEX_HISTORY_DB = "thread_history_1.sqlite"


def codex_db(name: str, home: Path | None = None):
    """Read-only connection to one Codex SQLite store (never written)."""
    import sqlite3
    p = (home or codex_home()) / name
    if not p.exists():
        return None
    return sqlite3.connect(f"file:{p.as_posix()}?mode=ro", uri=True)


#: Working folders of arena work: the hermetic player/gold folders (since 2026-10-18) and, before
#: containment, the repository itself (any checkout path, the Dropbox era included).
ARENA_CWD_PATTERNS = ("%sciencearena-hermetic%", "%ScienceArena%")


def codex_threads(home: Path | None = None, cwd_like: tuple[str, ...] = ARENA_CWD_PATTERNS) -> list[dict]:
    """Every Codex thread row whose working folder matches any pattern, as dicts."""
    con = codex_db(CODEX_STATE_DB, home)
    if con is None:
        return []
    import sqlite3
    con.row_factory = sqlite3.Row
    where = " or ".join("cwd like ?" for _ in cwd_like)
    rows = [dict(r) for r in con.execute(f"select * from threads where {where}", tuple(cwd_like))]
    con.close()
    return rows


def codex_thread_turns(thread_ids: list[str], home: Path | None = None) -> dict[str, list[dict]]:
    """Turns and items per thread id from the history store."""
    con = codex_db(CODEX_HISTORY_DB, home)
    out: dict[str, list[dict]] = {}
    if con is None:
        return out
    for tid in thread_ids:
        turns = []
        for status, err, started, completed, dur, turn_id in con.execute(
                "select status, error_json, started_at, completed_at, duration_ms, turn_id "
                "from thread_turns where thread_id=? order by rollout_ordinal", (tid,)):
            items = [(it, js) for it, js in con.execute(
                "select item_type, item_json from thread_items where thread_id=? and turn_id=? "
                "order by rollout_ordinal", (tid, turn_id))]
            turns.append({"status": status, "error_json": err, "started_at": started,
                          "completed_at": completed, "duration_ms": dur, "items": items})
        out[tid] = turns
    con.close()
    return out


def _codex_error_kind(error_json: str | None) -> str | None:
    if not error_json:
        return None
    try:
        e = json.loads(error_json)
    except json.JSONDecodeError:
        return "unparsed"
    info = e.get("codexErrorInfo") if isinstance(e, dict) else None
    if isinstance(info, str):
        return info
    if isinstance(info, dict) and info:
        return next(iter(info))
    return "error"


def final_codex_message_db(turns: list[dict]) -> str | None:
    """The last agent message of a thread from the history store (linking only; never stored)."""
    last = None
    for t in turns:
        for it, js in t["items"]:
            if it == "agentMessage":
                try:
                    last = json.loads(js).get("text", last)
                except json.JSONDecodeError:
                    pass
    return last


def _iso_ms(ms: int | None) -> str | None:
    if not ms:
        return None
    return _dt.datetime.fromtimestamp(ms / 1000, _dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def extract_codex_db(thread: dict, turns: list[dict]) -> dict:
    """Facts for one Codex session from its database rows (used when the rollout is gone,
    and as a cross-check when it is not). Fewer fields than a rollout: no request ids,
    no token split, no instructions digests."""
    sandbox = NOT_IN_LOG
    network = NOT_IN_LOG
    try:
        sp = json.loads(thread.get("sandbox_policy") or "")
        sandbox = sp.get("type", NOT_IN_LOG)
        if sp.get("type") == "managed":
            fs = sp.get("file_system") or {}
            sandbox = f"managed/{fs.get('type')}/" + ",".join(
                sorted({str(e.get("access")) for e in fs.get("entries") or []}))
        network = sp.get("network", NOT_IN_LOG)
    except (json.JSONDecodeError, AttributeError):
        pass
    called = []
    reasoning_items = 0
    for t in turns:
        for it, _ in t["items"]:
            if it == "reasoning":
                reasoning_items += 1
            elif it not in ("userMessage", "agentMessage"):
                called.append(it)
    eff = thread.get("reasoning_effort")
    ms0, ms1 = thread.get("created_at_ms"), thread.get("updated_at_ms")
    return {
        "facts_schema_version": FACTS_SCHEMA_VERSION,
        "source": "codex-state-db",
        "source_sha256": None,
        "harness": "codex-cli",
        "harness_version": thread.get("cli_version") or NOT_IN_LOG,
        "entrypoint": thread.get("source") or NOT_IN_LOG,
        "session_id": thread.get("id"),
        "model_requested": thread.get("model") or NOT_IN_LOG,
        "model_provider": thread.get("model_provider") or NOT_IN_LOG,
        # The rollouts that survive agree with the database on every session (checked by
        # scripts/recover_cli_provenance.py): NULL there is "absent" in turn_context.
        "effort_sent": eff if eff else ABSENT,
        "effort_reported": eff if eff else ABSENT,
        "sandbox": sandbox,
        "network": network,
        "approval_policy": thread.get("approval_mode") or NOT_IN_LOG,
        "tools_called": called,
        "reasoning_items": reasoning_items,
        "usage": {"total_tokens": thread.get("tokens_used")},
        "turns": [{"status": t["status"], "error": _codex_error_kind(t["error_json"]),
                   "duration_ms": t["duration_ms"]} for t in turns],
        "timing": {"turn_ms": sum(t["duration_ms"] or 0 for t in turns) or None},
        "first_timestamp": _iso_ms(ms0),
        "last_timestamp": _iso_ms(ms1),
    }


def final_codex_message(path: str | os.PathLike) -> str | None:
    """The last agent message of a rollout (used only to LINK a rollout to a record; never stored)."""
    last = None
    for o in _lines(path):
        p = o.get("payload") if isinstance(o.get("payload"), dict) else {}
        if o.get("type") == "event_msg" and p.get("type") == "task_complete":
            last = p.get("last_agent_message", last)
    return last


def codex_rollout_head(path: str | os.PathLike) -> dict:
    """session_meta + first turn_context of a rollout, cheaply (for indexing)."""
    out: dict = {}
    for i, o in enumerate(_lines(path)):
        p = o.get("payload") if isinstance(o.get("payload"), dict) else {}
        if o.get("type") == "session_meta":
            out["cwd"] = p.get("cwd")
            out["started"] = p.get("timestamp")
            out["session_id"] = p.get("session_id") or p.get("id")
        if o.get("type") == "turn_context":
            out["model"] = p.get("model")
            break
        if i > 40:
            break
    return out


# ---------------------------------------------------------------- held-out policy

#: Facts that may be published for a HELD-OUT record. Everything else (token
#: counts, timings, ids, request digests) can reveal the size or identity of a
#: held-out document and stays in article custody (same policy as
#: framework.holdout: held-out records drop `usage` entirely).
HELD_OUT_PUBLIC_FACTS = frozenset({
    "facts_schema_version", "source", "harness", "harness_version", "entrypoint",
    "model_requested", "models_served", "models_billed", "model_provider",
    "advisor_model_configured", "model_reroutes", "effort_sent", "effort_reported", "reasoning_summary",
    "thinking_type", "thinking_display", "thinking_budget_sent", "max_tokens_sent", "temperature_sent", "top_p_sent",
    "top_k_sent", "betas", "context_management", "cache_ttl", "system_prompt",
    "base_instructions", "model_instructions_hash", "model_catalog_hash", "tools_offered",
    "sandbox", "network", "approval_policy", "model_context_window", "service_tier",
    "inference_geo", "speed", "rate_limit_plan", "context_injections",
})


def public_view(facts: dict, *, held_out: bool) -> dict:
    """The part of `facts` that may be committed to this repository."""
    out = dict(facts)

    def strip(ci: dict) -> dict:
        ci = dict(ci or {})
        if held_out:
            ci.pop("host_skills_text", None)
        return ci
    ci = out.get("context_injections")
    # An aggregate over several sessions carries a list of distinct values.
    out["context_injections"] = [strip(c) for c in ci] if isinstance(ci, list) else strip(ci)
    if held_out:
        out = {k: v for k, v in out.items() if k in HELD_OUT_PUBLIC_FACTS}
        # Idempotent: a view of a view keeps the count instead of recounting an absent list.
        out["tools_called_count"] = (len(facts["tools_called"]) if isinstance(facts.get("tools_called"), list)
                                     else facts.get("tools_called_count", 0))
        out["withheld"] = "held-out: usage, timings, ids and digests kept in article custody"
    return out
