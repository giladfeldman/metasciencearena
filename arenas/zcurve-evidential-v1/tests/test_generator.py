"""Tests for the zcurve-evidential generator."""
import importlib.util
import sys
from collections import Counter
from pathlib import Path

ARENA_DIR = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("_zcurve_evidential_generator", ARENA_DIR / "generator.py")
generator = importlib.util.module_from_spec(_SPEC)
sys.modules["_zcurve_evidential_generator"] = generator
_SPEC.loader.exec_module(generator)

_Z_CRIT = 1.96


def test_generate_is_deterministic():
    a = list(generator.generate("v1", seed=3))
    b = list(generator.generate("v1", seed=3))
    assert [t["task_id"] for t in a] == [t["task_id"] for t in b]
    assert [t["input"]["z_scores"] for t in a] == [t["input"]["z_scores"] for t in b]


def test_envelopes_have_required_fields_and_tags():
    tasks = list(generator.generate("v1", seed=0, split="revealed"))
    assert tasks
    for t in tasks:
        assert {"task_id", "arena_id", "task_set_version", "difficulty", "input"} <= t.keys()
        assert t["arena_id"] == "zcurve-evidential-v1"
        assert t["split"] == "revealed"
        assert t["visibility"] == "public"
        assert {"study_set_id", "z_scores", "n_studies"} <= t["input"].keys()
        assert {"tier", "regime"} <= t["difficulty"].keys()


def test_all_z_scores_are_significant():
    for t in generator.generate("v1", seed=0):
        zs = t["input"]["z_scores"]
        assert zs and all(z > _Z_CRIT for z in zs)
        assert t["input"]["n_studies"] == len(zs)


def test_private_split_is_held_out():
    tasks = list(generator.generate("v1", seed=9, split="private"))
    assert all(t["split"] == "private" and t["visibility"] == "held_out" for t in tasks)


def test_all_six_tiers_present():
    tiers = {t["difficulty"]["tier"] for t in generator.generate("v1", seed=0)}
    assert tiers == {1, 2, 3, 4, 5, 6}


def test_ground_truth_returns_verdict_regime_and_mistake_kinds():
    tasks = list(generator.generate("v1", seed=0))
    gt = generator.ground_truth(tasks[0]["task_id"])
    assert "has_evidential_value" in gt and isinstance(gt["has_evidential_value"], bool)
    assert gt["regime"] in ("evidential", "non_evidential")
    assert isinstance(gt["mistake_kinds"], list) and gt["mistake_kinds"] == [gt["regime"]]


def test_ground_truth_missing_task_raises_keyerror():
    import pytest
    list(generator.generate("v1", seed=0))
    with pytest.raises(KeyError):
        generator.ground_truth("zc-t1-0-sDOES_NOT_EXIST")


def test_regime_matches_verdict():
    for t in generator.generate("v1", seed=0):
        gt = generator.ground_truth(t["task_id"])
        assert gt["has_evidential_value"] == (gt["regime"] == "evidential")
        assert gt["regime"] == t["difficulty"]["regime"]


def test_revealed_split_covers_both_regimes():
    """Parity needs BOTH regimes present in the revealed split -> 2 categories."""
    seen = set()
    for t in generator.generate("v1", seed=0, split="revealed"):
        seen.add(generator.ground_truth(t["task_id"])["regime"])
    assert seen == {"evidential", "non_evidential"}


def test_private_split_covers_both_regimes():
    seen = set()
    for t in generator.generate("v1", seed=12345, split="private"):
        seen.add(generator.ground_truth(t["task_id"])["regime"])
    assert seen == {"evidential", "non_evidential"}


def test_splits_share_tier_matrix_but_differ_in_content():
    rev = list(generator.generate("v1", seed=0, split="revealed"))
    priv = list(generator.generate("v1", seed=12345, split="private"))

    def cells(tasks):
        return Counter(t["difficulty"]["tier"] for t in tasks)

    assert cells(rev) == cells(priv)
    assert [t["input"]["z_scores"] for t in rev] != [t["input"]["z_scores"] for t in priv]
