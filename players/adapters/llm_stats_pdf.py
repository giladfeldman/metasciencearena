"""Output coercer for the stats-from-pdf-v1 API players (anthropic-api-*-statspdf, 2026-10-28).

The model reads the PDF and answers in JSON (arenas/stats-from-pdf-v1/prompts/v1.md).
This maps that answer onto the arena's output schema (schemas/output.schema.json), which
the runner enforces: an answer the schema rejects scores 0. It is format only -- it never
adds a result, and a value it cannot read as the schema's type becomes null, never a guess:
  - numbers given as strings ("−2.10", ".041", "0,45") are read as numbers, with the
    same comma rule as the arena scorer's own reader;
  - p_relation keeps only "=", "<" and ">";
  - verdict keeps the schema's four values, with "ok" read as "consistent" (the
    stats-extraction-v1 word models carry over); any other wording becomes null, which
    the scorer counts as no verdict (a non-hit, not a pass);
  - a result with no test_type string is dropped, as the scorer cannot match it.
"""
from __future__ import annotations

import json
import math
import re

_NUMBER = re.compile(r"^[+-]?(\d+([.,]\d*)?|[.,]\d+)$")
_NUM_FIELDS = ("df1", "df2", "statistic", "p_reported", "effect_size", "ci_low", "ci_high")
_RELATIONS = ("=", "<", ">")
_VERDICTS = {"consistent": "consistent", "ok": "consistent", "inconsistent": "inconsistent",
             "decision_error": "decision_error", "unverifiable": "unverifiable"}
_MAX_RESULTS = 1000


def _num(x):
    if isinstance(x, bool) or x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x) if math.isfinite(x) else None
    if isinstance(x, str):
        s = x.strip().replace("−", "-").replace("–", "-")
        if _NUMBER.match(s):
            return float(s.replace(",", "."))
    return None


def _page(x):
    v = _num(x)
    return int(v) if v is not None and v == int(v) and v >= 1 else None


def _str(x):
    return x if isinstance(x, str) else None


def _load(raw: str):
    s = (raw or "").strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", s)
    if fence:
        s = fence.group(1).strip()
    try:
        return json.loads(s)
    except ValueError:
        m = re.search(r"\{[\s\S]*\}", s)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except ValueError:
            return None


def coerce_stats_pdf(raw: str) -> dict:
    data = _load(raw)
    items = data.get("results") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return {"results": []}
    out = []
    for it in items[:_MAX_RESULTS]:
        if not isinstance(it, dict) or not isinstance(it.get("test_type"), str) or not it["test_type"].strip():
            continue
        rec = {"test_type": it["test_type"].strip()}
        for k in _NUM_FIELDS:
            rec[k] = _num(it.get(k))
        rel = it.get("p_relation")
        rec["p_relation"] = rel if rel in _RELATIONS else None
        v = it.get("verdict")
        rec["verdict"] = _VERDICTS.get(v.strip().lower()) if isinstance(v, str) else None
        rec["effect_size_type"] = _str(it.get("effect_size_type"))
        rec["raw"] = _str(it.get("raw"))
        rec["page"] = _page(it.get("page"))
        out.append(rec)
    answer = {"results": out}
    if isinstance(data.get("player_strategy_notes"), str):
        answer["player_strategy_notes"] = data["player_strategy_notes"]
    return answer
