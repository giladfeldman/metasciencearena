"""Human evaluators are scored by the arena's own scorer through the ordinary runner.

Two-sided: a perfect answer must score 1.0 and a wrong one must score lower,
through the same ``tools/human_score.py`` path production uses -- so a green
here cannot come from a scorer that returns a constant. And an answer to content
the person did not see must produce an ERROR record, never a number.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from players.adapters.human_replay import canonical_input_sha256

REPO = Path(__file__).resolve().parents[3]
ARENA = REPO / "arenas" / "prereg-deviation-v1"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# tools/ (repo root) and leaderboard-app/ are private: the public mirror ships
# neither, so these skip there with the reason named, and run here.
_needs_human_score = pytest.mark.skipif(
    not (REPO / "tools" / "human_score.py").is_file(),
    reason="tools/human_score.py is private tooling, not in the public mirror",
)
_needs_app_fixture = pytest.mark.skipif(
    not (REPO / "leaderboard-app" / "lib" / "__tests__" / "fixtures"
         / "human-canonical-vectors.json").is_file(),
    reason="the leaderboard app's fixture is not in the public mirror",
)


@pytest.fixture(scope="module")
def arena():
    gen = _load(ARENA / "generator.py", "pd_gen_human_test")
    tasks = {e["task_id"]: e for e in gen.generate("v1", 0, split="revealed")}
    return gen, tasks


def _perfect(gold: dict) -> dict:
    return {"deviations": [
        {"dimension": g["dimension"], "deviation": g["deviation"],
         "deviation_kind": g["deviation_kind"], "confidence": 1.0}
        for g in gold["dimensions"]]}


def _all_wrong(gold: dict) -> dict:
    return {"deviations": [
        {"dimension": g["dimension"], "deviation": not g["deviation"],
         "deviation_kind": "outcome_switch" if not g["deviation"] else None, "confidence": 1.0}
        for g in gold["dimensions"]]}


def _records(out_dir: Path, slug: str) -> dict[str, dict]:
    lines = (out_dir / f"{slug}.jsonl").read_text(encoding="utf-8").splitlines()
    return {r["task_id"]: r for r in map(json.loads, lines)}


@_needs_human_score
def test_scored_by_the_arena_scorer_two_sided(arena, tmp_path):
    human_score = _load(REPO / "tools" / "human_score.py", "human_score_under_test")
    gen, tasks = arena
    t_dev = "pd-t3-0-s0"   # a single-deviation task
    t_trap = "pd-t2-0-s0"  # paraphrase trap: no deviation
    export = {"arena_id": "prereg-deviation-v1", "task_set_version": "v1", "evaluators": [
        {"player_slug": "human-aaaa", "responses": {
            t: {"answer": _perfect(gen.ground_truth(t)),
                "input_sha256": canonical_input_sha256(tasks[t]["input"])}
            for t in (t_dev, t_trap)}},
        {"player_slug": "human-bbbb", "responses": {
            t: {"answer": _all_wrong(gen.ground_truth(t)),
                "input_sha256": canonical_input_sha256(tasks[t]["input"])}
            for t in (t_dev, t_trap)}},
    ]}
    written = human_score.score_export(export, ARENA, tmp_path)
    assert written == {"human-aaaa": 2, "human-bbbb": 2}

    good = _records(tmp_path, "human-aaaa")
    bad = _records(tmp_path, "human-bbbb")
    for t in (t_dev, t_trap):
        assert good[t]["score"]["primary"] == pytest.approx(1.0)
        assert bad[t]["score"]["primary"] < 0.5
        rec = good[t]
        assert rec["player_type"] == "human-baseline"
        assert rec["split"] == "revealed"
        assert rec["task_visibility"] == "public"
        assert rec["instrument"]  # the standard instrument block is stamped
        assert rec["player_identity"]


@_needs_human_score
def test_answer_to_unseen_content_is_an_error_not_a_score(arena, tmp_path):
    human_score = _load(REPO / "tools" / "human_score.py", "human_score_under_test2")
    gen, tasks = arena
    t = "pd-t3-0-s0"
    export = {"arena_id": "prereg-deviation-v1", "task_set_version": "v1", "evaluators": [
        {"player_slug": "human-cccc", "responses": {
            t: {"answer": _perfect(gen.ground_truth(t)), "input_sha256": "0" * 64}}}]}
    human_score.score_export(export, ARENA, tmp_path)
    rec = _records(tmp_path, "human-cccc")[t]
    assert rec["score"]["primary"] == 0.0
    assert "ShownInputMismatch" in rec["score"]["breakdown"]["error"]


@_needs_human_score
def test_slug_must_be_a_human_instrument():
    human_score = _load(REPO / "tools" / "human_score.py", "human_score_under_test3")
    with pytest.raises(ValueError):
        human_score.registry_entry("claude-opus", Path("x.json"))
    with pytest.raises(ValueError):
        human_score.registry_entry("human-Bad Slug", Path("x.json"))


# Pinned cross-language vectors. The SAME values are asserted by
# leaderboard-app/lib/__tests__/human-canonical.test.ts against the TypeScript
# implementation; if either side's canonical form changes, one of the two fails.
CANONICAL_VECTORS = [
    ({"b": 1, "a": [3, "x"]}, None),
    ({"paper": "Wir haben 150 Teilnehmende rekrutiert.\nÄ ö ü — “quoted”", "dimensions": ["a", "b"]}, None),
    ({"nested": {"z": None, "y": True, "x": 0.5}, "tab": "a\tb", "ctl": "\u0001"}, None),
    # Notation traps: Python repr and JavaScript toString disagree on these.
    ({"n": [1.0, -0.0, 20.0, 1e-7, 2.5e-7, 0.000001, 1e21, 1.5e21, 123456789012345680000.0,
            1e16, 0.1, -3.25, 7, 0]}, None),
    # Key order: JavaScript sorts by UTF-16 code unit.
    ({"Z": 1, "a": 2, "é": 3, "_": 4}, None),
]


@_needs_app_fixture
def test_canonical_vectors_file_matches(tmp_path):
    fixture = REPO / "leaderboard-app" / "lib" / "__tests__" / "fixtures" / "human-canonical-vectors.json"
    pinned = json.loads(fixture.read_text(encoding="utf-8"))
    assert len(pinned) == len(CANONICAL_VECTORS)
    for (obj, _), row in zip(CANONICAL_VECTORS, pinned):
        assert row["input"] == obj
        assert canonical_input_sha256(obj) == row["sha256"]
