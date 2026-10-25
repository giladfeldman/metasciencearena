"""Measure a free model's real daily allowance, in arena-sized calls. NOT a scoring tool.

Owner decision 2026-10-15: citeclaim's default and this arena's primary target
move off Gemini Flash (20 requests/day on the free key) to a more generous free
model, chosen on MEASURED daily limits and accuracy. Provider pages are claims;
this probe is the measurement.

What it sends: only the arena's revealed split (the hand-written FICTIONAL
examples in catalogs/revealed_examples.yaml), rendered through the arena prompt
with the player's own settings, round-robin. No held-out text leaves this
machine, and nothing here is scored or written to a run record.

It calls until the provider answers with a per-DAY limit (an HTTP 429 whose
structured quota fields name a daily quota: classify_429), until --max-calls or
--max-hours, after 60 consecutive per-minute 429s or 10 consecutive 429s of
unknown kind, or after 5 consecutive non-429 failures. A per-minute 429 is
waited out and does not count. Every call's status, tokens, latency and
rate-limit headers, and every 429 body, go to the --out JSON (saved every 25
calls). The fictional examples are shorter than real tasks: compare the
recorded tokens_mean with a real run's before converting a token limit to tasks.

    python arenas/citation-claim-fidelity-v1/tools/free_quota_probe.py --player <registry id>
        --max-calls 1100 --out tmp/quota/<player>.json [--min-interval 0] [--max-hours 6]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml

ARENA = Path(__file__).resolve().parents[1]
REPO = ARENA.parents[1]
sys.path.insert(0, str(REPO))

GEMINI = "https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent"


def _entry(player_id: str) -> dict:
    reg = yaml.safe_load((REPO / "players" / "registry.yaml").read_text(encoding="utf-8"))
    rows = reg if isinstance(reg, list) else reg.get("players", [])
    for e in rows:
        if e.get("player_id") == player_id:
            return e
    raise SystemExit(f"{player_id} not in players/registry.yaml")


def _prompts() -> list[str]:
    import importlib.util
    spec = importlib.util.spec_from_file_location("ccf_gen_probe", ARENA / "generator.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    tpl = (ARENA / "prompts" / "v1.md").read_text(encoding="utf-8").replace(chr(13) + chr(10), chr(10))
    return [tpl.format(**env["input"]["prompt_fields"]) for env in gen.generate("v1", 0, split="revealed")]


def _request(e: dict, prompt: str) -> tuple[str, dict, dict]:
    """(url, headers, body) exactly as the player's adapter would send them."""
    if e["adapter_class"] == "GeminiGenerateContentAdapter":
        gen = {"temperature": e.get("gemini_temperature", 0), "maxOutputTokens": e.get("gemini_max_output_tokens", 8192)}
        if e.get("gemini_seed") is not None:
            gen["seed"] = e["gemini_seed"]
        if e.get("gemini_response_json", True):
            gen["responseMimeType"] = "application/json"
        if e.get("gemini_thinking_level"):
            gen["thinkingConfig"] = {"thinkingLevel": e["gemini_thinking_level"]}
        key = os.environ[e.get("gemini_api_key_env", "GEMINI_API_KEY")]
        return (GEMINI.format(m=e["gemini_model"]), {"x-goog-api-key": key},
                {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": gen})
    if e["adapter_class"] == "OpenAIChatCompletionsAdapter":
        # The body players/adapters/openai_compatible.py builds (consult 2026-10-15:
        # the first version omitted temperature, stream and the default ceiling).
        from players.adapters.openai_compatible import _DEFAULT_MAX_TOKENS
        base = os.environ[e["openai_base_url_env"]].rstrip("/")
        key = os.environ[e["openai_api_key_env"]]
        path = e.get("openai_chat_path", "/chat/completions")
        body = {"model": e["openai_model"], "temperature": e.get("openai_temperature", 0.0), "stream": False,
                "max_tokens": _DEFAULT_MAX_TOKENS, "messages": [{"role": "user", "content": prompt}]}
        body.update(e.get("openai_extra_body") or {})
        return base + (path if path.startswith("/") else "/" + path), {"Authorization": f"Bearer {key}"}, body
    raise SystemExit(f"adapter {e['adapter_class']} not supported by the probe")


def _tokens(data: dict) -> int | None:
    if "usageMetadata" in data:
        return data["usageMetadata"].get("totalTokenCount")
    return (data.get("usage") or {}).get("total_tokens")


def classify_429(text: str) -> str:
    """'daily', 'minute' or 'unknown', from the provider's STRUCTURED quota fields.

    Consult 2026-10-15: a free-text regex called a per-minute message that merely
    mentioned "daily" a daily limit, and missed "requests/day". Gemini names the
    violated quota (`quotaId` ...PerDay... / ...PerMinute...); Groq names the
    metric in parentheses ("(TPD)", "(RPM)"). Anything else is 'unknown', and an
    unknown limit is reported as unknown, never as a day's allowance.
    """
    try:
        err = (json.loads(text) or {}).get("error") or {}
    except (ValueError, AttributeError):
        return "unknown"
    ids = [v.get("quotaId", "") for d in err.get("details") or [] if isinstance(d, dict)
           for v in d.get("violations") or [] if isinstance(v, dict)]
    if any("PerDay" in q for q in ids):
        return "daily"
    if ids and all("PerMinute" in q for q in ids):
        return "minute"
    m = re.search(r"\((TPD|RPD|TPM|RPM)\)", str(err.get("message", "")))
    if m:
        return "daily" if m.group(1) in ("TPD", "RPD") else "minute"
    return "unknown"


def _retry_after(headers: dict) -> float:
    for k, v in headers.items():
        if k.lower() == "retry-after":
            try:
                return float(v)
            except ValueError:  # an HTTP-date: fall back to the default wait
                return 30.0
    return 30.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--player", required=True)
    ap.add_argument("--max-calls", type=int, default=1100)
    ap.add_argument("--max-hours", type=float, default=6.0, help="wall-clock cap (a quota day must not be spanned)")
    ap.add_argument("--min-interval", type=float, default=0.0, help="seconds between call starts")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    e = _entry(a.player)
    prompts = _prompts()
    calls, bodies_429, ok, fails_in_row, throttled_in_row, stop = [], [], 0, 0, 0, "max_calls"
    started, t_start = datetime.now(timezone.utc).isoformat(), time.time()

    def save(final: bool) -> dict:
        toks = [c["tokens"] for c in calls if c.get("tokens")]
        out = {"player": a.player, "model": e.get("gemini_model") or e.get("openai_model"),
               "started_utc": started, "ended_utc": datetime.now(timezone.utc).isoformat(),
               "stop_reason": stop if final else "running", "successful_calls": ok, "attempts": len(calls),
               "n_429": len(bodies_429), "tokens_total": sum(toks),
               "tokens_mean": round(sum(toks) / len(toks)) if toks else None,
               "served_models": sorted({c["served"] for c in calls if c.get("served")}),
               "last_ratelimit_headers": next((c["ratelimit_headers"] for c in reversed(calls)
                                               if c["ratelimit_headers"]), {}),
               "bodies_429": bodies_429[-5:], "first_429": bodies_429[:1], "calls": calls}
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
        return out

    i = 0
    while ok < a.max_calls:
        if time.time() - t_start > a.max_hours * 3600:
            stop = "max_hours"
            break
        url, headers, body = _request(e, prompts[i % len(prompts)])
        t0 = time.time()
        r = None
        try:
            r = requests.post(url, json=body, headers={**headers, "Content-Type": "application/json"}, timeout=300)
            status, text, rh = r.status_code, r.text, {k: v for k, v in r.headers.items() if "ratelimit" in k.lower()
                                                       or k.lower() == "retry-after"}
        except requests.RequestException as ex:
            status, text, rh = -1, type(ex).__name__, {}
        dt = round(time.time() - t0, 2)
        rec = {"t": datetime.now(timezone.utc).isoformat(timespec="seconds"), "status": status, "latency_s": dt,
               "ratelimit_headers": rh}
        data = None
        if status == 200:
            try:
                data = r.json()
            except ValueError:
                status = rec["status"] = -2  # a 200 that is not JSON is a failure, not a success
        if status == 200:
            rec["tokens"] = _tokens(data)
            rec["served"] = data.get("modelVersion") or data.get("model")
            ok += 1
            fails_in_row = throttled_in_row = 0
            i += 1
        elif status == 429:
            kind = classify_429(text)
            rec["limit"] = kind
            bodies_429.append({"t": rec["t"], "limit": kind, "body": text[:2000], "headers": rh})
            throttled_in_row += 1
            if kind == "daily":
                calls.append(rec)
                stop = "daily_limit"
                break
            if throttled_in_row >= (10 if kind == "unknown" else 60):
                calls.append(rec)
                stop = f"repeated_{kind}_429"
                break
            time.sleep(min(max(_retry_after(rh), 5.0), 120.0))
        else:
            fails_in_row += 1
            rec["body"] = text[:400]
            if fails_in_row >= 5:
                calls.append(rec)
                stop = "repeated_errors"
                break
            time.sleep(10)
        calls.append(rec)
        if a.min_interval:
            time.sleep(max(0.0, a.min_interval - dt))
        if len(calls) % 25 == 0:
            save(final=False)
            print(f"{a.player}: ok={ok} calls={len(calls)} 429s={len(bodies_429)}", flush=True)
    out = save(final=True)
    print(json.dumps({k: out[k] for k in ("player", "stop_reason", "successful_calls", "n_429", "tokens_mean",
                                          "served_models", "last_ratelimit_headers")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
