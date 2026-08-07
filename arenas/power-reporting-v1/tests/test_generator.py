"""Tests for the power-reporting generator."""
import importlib.util
import sys
from collections import Counter
from pathlib import Path

ARENA_DIR = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("_power_reporting_generator", ARENA_DIR / "generator.py")
generator = importlib.util.module_from_spec(_SPEC)
sys.modules["_power_reporting_generator"] = generator
_SPEC.loader.exec_module(generator)


def test_generate_is_deterministic():
    a = list(generator.generate("v1", seed=3))
    b = list(generator.generate("v1", seed=3))
    assert [t["task_id"] for t in a] == [t["task_id"] for t in b]
    assert [t["input"]["text"] for t in a] == [t["input"]["text"] for t in b]


def test_envelopes_have_required_fields_and_tags():
    tasks = list(generator.generate("v1", seed=0, split="revealed"))
    assert tasks
    for t in tasks:
        assert {"task_id", "arena_id", "task_set_version", "difficulty", "input"} <= t.keys()
        assert t["arena_id"] == "power-reporting-v1"
        assert t["split"] == "revealed"
        assert t["visibility"] == "public"
        assert "text" in t["input"]
        assert isinstance(t["input"]["text"], str) and t["input"]["text"]


def test_private_split_is_held_out():
    tasks = list(generator.generate("v1", seed=9, split="private"))
    assert tasks
    assert all(t["split"] == "private" and t["visibility"] == "held_out" for t in tasks)


def test_all_six_tiers_present():
    tiers = {t["difficulty"]["tier"] for t in generator.generate("v1", seed=0)}
    assert tiers == {1, 2, 3, 4, 5, 6}


def test_task_count_matches_manifest():
    # 21 after the 2026-06-29 broaden added sensitivity_as_apriori (T3 + T5 each
    # cycle MISTAKE_KINDS, so +1 kind ⇒ +2 tasks vs the original 19). Keep in sync
    # with arena.yaml task_set_versions[].n_tasks.
    tasks = list(generator.generate("v1", seed=0))
    assert len(tasks) == 21


def test_ground_truth_returns_field_map_shape():
    tasks = list(generator.generate("v1", seed=0))
    gt = generator.ground_truth(tasks[0]["task_id"])
    assert {"has_power_analysis", "kind", "fields", "mistake_kinds"} <= gt.keys()
    assert isinstance(gt["fields"], dict)
    assert isinstance(gt["mistake_kinds"], list)


def test_ground_truth_raises_on_unknown_task():
    import pytest
    with pytest.raises(KeyError):
        generator.ground_truth("pr-does-not-exist-s0")


def test_every_gold_declares_mistake_kinds_list():
    for t in generator.generate("v1", seed=0):
        gt = generator.ground_truth(t["task_id"])
        assert isinstance(gt["mistake_kinds"], list)


def test_revealed_set_covers_every_mistake_kind_and_clean():
    """The public benchmark must exercise the full array of injected mistakes."""
    seen: set[str] = set()
    saw_clean = False
    for t in generator.generate("v1", seed=0, split="revealed"):
        gt = generator.ground_truth(t["task_id"])
        ks = gt["mistake_kinds"]
        if not ks:
            saw_clean = True
        seen.update(ks)
    assert set(generator.MISTAKE_KINDS) <= seen
    assert saw_clean


def test_no_power_analysis_gold_is_consistent():
    for t in generator.generate("v1", seed=0):
        gt = generator.ground_truth(t["task_id"])
        if "no_power_analysis" in gt["mistake_kinds"]:
            assert gt["has_power_analysis"] is False
            assert gt["kind"] is None
            assert gt["fields"] == {}


def test_sensitivity_as_apriori_is_injected_and_true_kind_is_sensitivity():
    """2026-06-29 broaden (cycle 2): a sensitivity analysis worded as a-priori. The
    mistake is in MISTAKE_KINDS, appears in revealed gold, its TRUE kind is
    'sensitivity' (not apriori), and it keeps a full field set — and the legitimate
    look-alike (clean `sensitivity`, in CLEAN_KINDS) is the confusable control."""
    assert "sensitivity_as_apriori" in generator.MISTAKE_KINDS
    assert "sensitivity" in generator.CLEAN_KINDS  # the confusable clean control exists
    seen = False
    for t in generator.generate("v1", seed=0, split="revealed"):
        gt = generator.ground_truth(t["task_id"])
        if "sensitivity_as_apriori" in gt["mistake_kinds"]:
            seen = True
            assert gt["has_power_analysis"] is True
            assert gt["kind"] == "sensitivity", "true kind must be sensitivity, not apriori"
            assert set(gt["fields"]) == set(generator.FIELDS), "all fields present, just mislabelled"
    assert seen, "sensitivity_as_apriori never appeared in revealed gold"


def test_posthoc_as_apriori_gold_true_kind_is_posthoc():
    for t in generator.generate("v1", seed=0):
        gt = generator.ground_truth(t["task_id"])
        if "posthoc_as_apriori" in gt["mistake_kinds"]:
            assert gt["has_power_analysis"] is True
            assert gt["kind"] == "posthoc"


def test_missing_fields_gold_omits_some_fields():
    found = False
    for t in generator.generate("v1", seed=0):
        gt = generator.ground_truth(t["task_id"])
        if "missing_fields" in gt["mistake_kinds"]:
            found = True
            assert gt["has_power_analysis"] is True
            assert 0 < len(gt["fields"]) < len(generator.FIELDS)
    assert found


def test_clean_tasks_have_all_fields():
    for t in generator.generate("v1", seed=0):
        gt = generator.ground_truth(t["task_id"])
        if not gt["mistake_kinds"] and gt["has_power_analysis"]:
            assert set(gt["fields"]) == set(generator.FIELDS)


def test_splits_share_difficulty_matrix_but_differ_in_content():
    rev = list(generator.generate("v1", seed=0, split="revealed"))
    priv = list(generator.generate("v1", seed=12345, split="private"))

    def cells(tasks):
        return Counter(t["difficulty"]["tier"] for t in tasks)

    assert cells(rev) == cells(priv)
    assert [t["input"]["text"] for t in rev] != [t["input"]["text"] for t in priv]
