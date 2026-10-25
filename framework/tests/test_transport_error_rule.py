"""A TRANSPORT ERROR (HTTP 429 / 5xx / connection) is the provider's, never the model's answer.

d-685d09 (2026-10-19). `holdout.sanitize_error` reduced a held-out record's error to
its exception TYPE, so `RuntimeError: <player>: ... returned HTTP 429: ...` became
`RuntimeError`, and two consumers then mistook a quota failure for the model's own:
`scripts/run_health.py` filed it as errored_deterministic, and the board's
unscoreable rule zeroed it. Measured that day: gpt-oss-120b lost 306 of 419 calls and
Flash-Lite 93 of 589 to free-tier quotas, every one reduced to `RuntimeError`.

The fix mirrors the TransportTimeout precedent (d-545c62): keep the CLASS, not the
text, and exclude it from scoring the way a timeout is excluded. The cases live in
`fixtures/transport_error_cases.json`, which the Node parity test reads too.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from framework import timeouts
from framework.holdout import sanitize_error
from framework.report import UNSCOREABLE_ZERO_FLAG, apply_unscoreable_policy

REPO = Path(__file__).resolve().parents[2]
CASES = json.loads((Path(__file__).parent / "fixtures" / "transport_error_cases.json")
                   .read_text(encoding="utf-8"))["cases"]


@pytest.mark.parametrize("case", CASES, ids=[c["why"][:40] for c in CASES])
def test_classifier_matches_the_shared_case_table(case):
    assert timeouts.is_transport_error(case["error"]) is case["transport"], case["why"]


@pytest.mark.parametrize("case", CASES, ids=[c["why"][:40] for c in CASES])
def test_sanitize_keeps_the_transport_class(case):
    out = sanitize_error(case["error"])
    assert out == case["sanitized"], case["why"]
    # The redacted form must still classify the same way: that is the whole point.
    assert timeouts.is_transport_error(out) is case["transport"], case["why"]


def test_a_timeout_is_not_reclassified_as_a_transport_error():
    # The two classes stay apart: C11 blocks timeouts at ZERO tolerance, C7 allows
    # 10 % transport errors. A timeout must keep its own name.
    msg = "RuntimeError: claude exited 124: <stderr>"
    assert sanitize_error(msg) == "TransportTimeout"
    assert not timeouts.is_transport_error(msg)
    assert timeouts.is_transport_failure(msg)


def _rec(task, player, error=None, primary=0.5):
    bd = {"error": error} if error else {}
    return {"task_id": task, "player_id": player,
            "score": {"primary": None if error else primary, "breakdown": bd}}


def test_policy_never_zeroes_a_held_out_quota_failure():
    recs = [_rec("t1", "a"), _rec("t1", "b", "TransportError"),
            _rec("t2", "b", "TransportError")]
    out, orphans = apply_unscoreable_policy(recs)
    b = [r for r in out if r["player_id"] == "b"]
    assert all(r["score"]["primary"] is None for r in b)
    assert not any(r["score"]["breakdown"].get(UNSCOREABLE_ZERO_FLAG) for r in b)
    # Nobody scored t2, but a quota failure says nothing about the TASK.
    assert orphans == set()


def test_policy_still_zeroes_the_models_own_failure():
    out, _ = apply_unscoreable_policy([_rec("t1", "a"), _rec("t1", "b", "RuntimeError")])
    assert [r for r in out if r["player_id"] == "b"][0]["score"]["primary"] == 0.0


def _load(name: str, rel: str):
    path = REPO / rel
    if not path.is_file():
        pytest.skip(f"{rel} is not published - private checkout only")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    # Registered before exec: @dataclass resolves annotations via sys.modules.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("error", [
    "TransportError",
    "HTTPError: 503 Server Error: Service Unavailable for url: https://example.invalid",
])
def test_run_health_files_a_quota_failure_as_transport(tmp_path, error):
    rh = _load("run_health_te", "scripts/run_health.py")
    recs = [{"task_id": f"t{i}", "player_id": "p", "score": {"primary": None,
             "breakdown": {"error": error}}} for i in range(3)]
    p = tmp_path / "p__private__r1.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
    kinds = {i.kind for i in rh.check(p)}
    assert "errored_transport" in kinds and "errored_deterministic" not in kinds


def test_c7_counts_a_held_out_quota_failure(tmp_path, monkeypatch):
    qa = _load("qa_cov_te", "scripts/qa_coverage_checks.py")
    d = tmp_path / "arenas" / "fake-arena-v1" / "runs" / "v1"
    d.mkdir(parents=True)
    recs = [{"task_id": f"t{i}", "player_id": "p", "score": {"primary": None,
             "breakdown": {"error": "TransportError"}}} for i in range(5)]
    (d / "p__private__r1.jsonl").write_text(
        "\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
    monkeypatch.setattr(qa, "ARENAS", tmp_path / "arenas")
    fails = qa.check_transport_error_concentration()
    assert len(fails) == 1 and "5/5" in fails[0]
