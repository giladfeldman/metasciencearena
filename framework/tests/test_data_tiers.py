"""The data-tier rule (framework/data_tiers.py), pinned to the fixture the Node build shares.

leaderboard-app/scripts/tests/data-tiers.test.mjs checks the Node mirror against the same
``framework/contract/fixtures/data_tier_cases.json``, so the board and the Python report agree
on which tasks form a tier, who is complete on it and what it scored.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from framework.data_tiers import parse_tier_index, plan_data_tiers, public_tier_plan, read_tier_index

REPO = Path(__file__).resolve().parents[2]
FIXTURE = json.loads((REPO / "framework" / "contract" / "fixtures" / "data_tier_cases.json").read_text(encoding="utf-8"))
CCF = REPO / "arenas" / "citation-claim-fidelity-v1"


def _index(rows):
    if rows is None:
        return None
    return parse_tier_index("\n".join(json.dumps(r) for r in rows))


def _same(got, want, path="$"):
    if isinstance(want, float) or isinstance(got, float):
        assert got is not None and want is not None and math.isclose(got, want, abs_tol=1e-12), f"{path}: {got} != {want}"
    elif isinstance(want, dict):
        assert isinstance(got, dict) and set(got) == set(want), f"{path}: keys {sorted(got or {})} != {sorted(want)}"
        for k in want:
            _same(got[k], want[k], f"{path}.{k}")
    elif isinstance(want, list):
        assert isinstance(got, list) and len(got) == len(want), f"{path}: {got} != {want}"
        for i, (g, w) in enumerate(zip(got, want)):
            _same(g, w, f"{path}[{i}]")
    else:
        assert got == want, f"{path}: {got!r} != {want!r}"


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda c: c["name"])
def test_plan_matches_the_shared_fixture(case):
    plan = plan_data_tiers(tier_index=_index(case["tier_index"]), declarations=case["declarations"],
                           records=case["records"])
    _same(public_tier_plan(plan), case["expected"])


def test_a_tier_without_a_score_label_never_yields_a_mean():
    case = FIXTURE["cases"][0]
    plan = plan_data_tiers(tier_index=_index(case["tier_index"]), declarations=case["declarations"],
                           records=case["records"])
    zhang = next(t for t in plan["tiers"] if t["id"] == "rd-zhang2024")
    assert zhang["players"], "the fixture must give the Zhang tier a complete player, or this is a false green"
    for row in zhang["players"]:
        assert row["complete"] and "mean" not in row and "n_scored" not in row


def test_tiers_are_never_pooled():
    # Every task belongs to exactly one tier, so a player's tier means are over disjoint tasks.
    case = FIXTURE["cases"][0]
    plan = plan_data_tiers(tier_index=_index(case["tier_index"]), declarations=case["declarations"],
                           records=case["records"])
    seen: set[str] = set()
    for t in plan["tiers"]:
        assert not (seen & set(t["task_ids"]))
        seen |= set(t["task_ids"])
    a = {t["id"]: next(p for p in t["players"] if p["player_id"] == "A") for t in plan["tiers"]}
    pooled = (1.0 + 0.5 + 0.0 + 0.5 + 1.0) / 5   # A's synthetic + mmc tasks averaged together
    assert a["synthetic"]["mean"] != pooled and a["rd-mmc-pilot"]["mean"] != pooled


# ccf's task sets hold real-paper ground truth and stay private; the public mirror ships no tiers.jsonl.
@pytest.mark.skipif(not (CCF / "task_sets" / "v3" / "tiers.jsonl").exists(),
                    reason="citation-claim-fidelity-v1 task sets are private, not in the public mirror")
def test_ccf_declares_every_tier_its_tier_index_names():
    import yaml
    decl = yaml.safe_load((CCF / "arena.yaml").read_text(encoding="utf-8"))["data_tiers"]
    assert decl["rd-zhang2024"]["score_label"] is None
    for v in ("v3", "v4"):
        idx = read_tier_index(CCF, v)
        assert idx, f"{v}: no tiers.jsonl"
        plan = plan_data_tiers(tier_index=idx, declarations=decl, records=[])
        assert plan["undeclared_tiers"] == []
    assert read_tier_index(CCF, "v2") is None   # v2 is one population and stays untouched


def _write_tiered_arena(tmp_path: Path) -> Path:
    """A copy of the fixture as a tiny on-disk arena, for the Python report engine."""
    import shutil

    import yaml

    case = FIXTURE["cases"][0]
    arena = tmp_path / "citation-claim-fidelity-v1"
    shutil.copytree(CCF, arena, ignore=shutil.ignore_patterns("runs", "task_sets", "__pycache__", "tests"))
    m = yaml.safe_load((arena / "arena.yaml").read_text(encoding="utf-8"))
    m["data_tiers"] = case["declarations"]
    (arena / "arena.yaml").write_text(yaml.safe_dump(m, sort_keys=False), encoding="utf-8")
    ts = arena / "task_sets" / "v9"
    ts.mkdir(parents=True)
    rows = [r for r in case["tier_index"] if r["tier"] != "extra"]
    (ts / "tiers.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    runs = arena / "runs" / "v9"
    runs.mkdir(parents=True)
    recs = [dict(r, task_set_version="v9") for r in case["records"] if r["task_id"] in {x["task_id"] for x in rows}]
    (runs / "fixture.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    return arena


def test_python_report_scores_one_tier_and_refuses_pooling_and_zhang(tmp_path):
    from framework.report import generate_report

    arena = _write_tiered_arena(tmp_path)
    kw = dict(arena_dir=arena, task_set_version="v9", player_id="A", player_version="A-1")
    with pytest.raises(ValueError, match="never pooled"):
        generate_report(**kw)
    with pytest.raises(ValueError, match="publishes no score"):
        generate_report(**kw, tier="rd-zhang2024")
    bundle = generate_report(**kw, tier="synthetic")
    report = json.loads((bundle / "tool_report.json").read_text(encoding="utf-8"))
    want = next(p for p in FIXTURE["cases"][0]["expected"]["tiers"][0]["players"] if p["player_id"] == "A")
    assert report["data_tier"]["mean"] == pytest.approx(want["mean"], abs=1e-12)
    # The report's own summary is computed over the same tier records, so it agrees too.
    assert report["summary"]["primary_mean"] == pytest.approx(want["mean"], abs=1e-12)
    with pytest.raises(ValueError, match="did not run every task"):
        generate_report(arena_dir=arena, task_set_version="v9", player_id="C", player_version="C-1", tier="synthetic")
