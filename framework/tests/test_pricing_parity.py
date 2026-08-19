"""The Python and JavaScript cost engines must agree, field by field.

There are two on purpose: `framework/pricing.py` is the reference and the local
CLI (`framework report`, and the wheel a stranger `pip install`s), while
`leaderboard-app/scripts/lib/pricing.mjs` runs inside the Node build Vercel
actually executes to publish the site.

Two implementations invite drift, and the drift here does not crash — it
publishes a different DOLLAR or TOKEN figure under the same field name. That is
the exact failure this project has already shipped once (`cost_usd` null in all
127 reports, 2026-08-14), and the exact class the Fable 5 cross-review found
again on 2026-08-15: the Node engine had received a strict-integer fix from a
Codex review that was never mirrored into Python, and Python returned raw sums
where Node returned null.

The previous "parity" tests restated ONE side's expectations in that side's own
language, which cannot catch a divergence. This one runs BOTH engines over ONE
shared fixture and compares the outputs, including the edge cases where the two
languages disagree by default: `{}` (falsy in Python, truthy in JS), booleans
(`isinstance(True, int)` is True in Python, `Number.isInteger(true)` is false),
and non-integer counts.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from framework import pricing

REPO = Path(__file__).resolve().parents[2]
JS_MODULE = REPO / "leaderboard-app" / "scripts" / "lib" / "pricing.mjs"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None or not JS_MODULE.exists(),
    reason="node and leaderboard-app/scripts/lib/pricing.mjs are both required",
)

#: The keys mean the same thing on both sides; only the spelling differs.
#: (Python is snake-noun-first, the published JSON is `tokens_*`.)
_KEY_MAP = {
    "prompt_tokens": "tokens_prompt",
    "completion_tokens": "tokens_completion",
    "total_tokens": "tokens_total",
    "n_with_usage": "n_with_usage",
    "n_priced": "n_priced",
    "usage_absent_reason": "usage_absent_reason",
    "price_table_checked_on": "price_table_checked_on",
    "price_table_stale": "price_table_stale",
}

_PRICED_MODEL = "openai/gpt-oss-120b"


def _rec(**kw) -> dict:
    base = {
        "player_id": "probe",
        "player_type": "model",
        "timestamp_utc": "2026-08-14T10:00:00Z",
        "resolved_tool_version": _PRICED_MODEL,
    }
    base.update(kw)
    return base


#: One case per way the two languages can silently part company.
CASES: dict[str, list[dict]] = {
    "no_usage_at_all": [_rec()],
    "subscription_player": [
        _rec(player_id="claude-opus-5-x", player_version="claude-opus-5",
             resolved_tool_version=None)
    ],
    "local_tool_player": [_rec(player_id="statcheck", player_type="tool",
                               resolved_tool_version=None)],
    "empty_usage_block": [_rec(usage={})],
    "float_prompt_tokens": [_rec(usage={"prompt_tokens": 780.5, "completion_tokens": 212})],
    "boolean_tokens": [_rec(usage={"prompt_tokens": True, "completion_tokens": True})],
    "string_tokens": [_rec(usage={"prompt_tokens": "780", "completion_tokens": 212})],
    "null_completion": [_rec(usage={"prompt_tokens": 780, "completion_tokens": None})],
    "well_formed": [_rec(usage={"prompt_tokens": 780, "completion_tokens": 212})],
    "unpriced_model": [_rec(resolved_tool_version="not-in-the-table",
                            usage={"prompt_tokens": 780, "completion_tokens": 212})],
    "mixed_batch": [
        _rec(usage={"prompt_tokens": 780, "completion_tokens": 212}),
        _rec(usage={"prompt_tokens": 900, "completion_tokens": 300}),
        _rec(player_id="statcheck", player_type="tool", resolved_tool_version=None),
    ],
    "pre_instrument": [_rec(timestamp_utc="2026-07-01T10:00:00Z", resolved_tool_version=None)],
}


def _js_summarise(cases: dict[str, list[dict]]) -> dict[str, dict]:
    """Run summariseUsage() over every case in one Node process."""
    # The module specifier must be a file:// URL — a Windows absolute path is not
    # a supported ESM scheme ("Received protocol 'c:'").
    module_url = JS_MODULE.resolve().as_uri()
    script = f"""
import {{ loadPriceTable, summariseUsage }} from {json.dumps(module_url)};
const t = loadPriceTable({json.dumps(REPO.as_posix())});
const cases = {json.dumps(cases)};
const out = {{}};
for (const [name, records] of Object.entries(cases)) out[name] = summariseUsage(t, records);
process.stdout.write(JSON.stringify(out));
"""
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        capture_output=True, text=True, cwd=REPO,
    )
    assert proc.returncode == 0, f"node failed: {proc.stderr}"
    return json.loads(proc.stdout)


def test_both_engines_agree_on_every_edge_case():
    """Field-by-field equality over a shared fixture, in both directions.

    A divergence here means the CLI a stranger installs and the site a funder
    reads publish different numbers under the same names.
    """
    js = _js_summarise(CASES)
    assert set(js) == set(CASES), "the Node run did not cover every case"

    mismatches: list[str] = []
    for name, records in CASES.items():
        py = pricing.summarise(records)
        for py_key, js_key in _KEY_MAP.items():
            got_py, got_js = py[py_key], js[name][js_key]
            if got_py != got_js:
                mismatches.append(
                    f"{name}.{py_key}: python={got_py!r} node={got_js!r} (as {js_key})"
                )
    assert not mismatches, "the two cost engines disagree:\n  " + "\n  ".join(mismatches)


def test_the_fixture_actually_exercised_both_engines():
    """A parity test over an empty case set passes while proving nothing.

    This project's own rule: assert the input was non-empty before trusting any
    all-clear. (LESSONS.md, 'a green result from an empty input is a false
    green'.)
    """
    assert len(CASES) >= 10, "too few parity cases to be meaningful"
    js = _js_summarise(CASES)
    assert any(v["n_with_usage"] > 0 for v in js.values()), "no case carried usage"
    assert any(v["n_priced"] > 0 for v in js.values()), "no case was ever priced"
    assert any(v["usage_absent_reason"] for v in js.values()), "no absence case"


def test_cost_totals_agree_between_engines():
    """The dollar figure itself, not just the token counts.

    Python reports a TOTAL (`cost_usd`) and Node a MEAN (`cost_usd_mean`) by
    design; this derives one from the other so a pricing-table or rounding
    divergence still fails.
    """
    js = _js_summarise(CASES)
    for name, records in CASES.items():
        py = pricing.summarise(records)
        py_total, n_priced = py["cost_usd"], py["n_priced"]
        js_mean = js[name]["cost_usd_mean"]
        if n_priced == 0:
            assert py_total is None and js_mean is None, f"{name}: unpriced must be null both sides"
            continue
        assert js_mean is not None, f"{name}: node dropped a priced record"
        assert py_total == pytest.approx(js_mean * n_priced, rel=1e-9), (
            f"{name}: python total {py_total} != node mean {js_mean} x {n_priced}"
        )
