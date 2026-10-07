"""Tests for the prereg-deviation generator."""
import importlib.util
import sys
from collections import Counter
from pathlib import Path

ARENA_DIR = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("_prereg_deviation_generator", ARENA_DIR / "generator.py")
generator = importlib.util.module_from_spec(_SPEC)
sys.modules["_prereg_deviation_generator"] = generator
_SPEC.loader.exec_module(generator)


def test_generate_is_deterministic():
    a = list(generator.generate("v1", seed=3))
    b = list(generator.generate("v1", seed=3))
    assert [t["task_id"] for t in a] == [t["task_id"] for t in b]
    assert [t["input"]["paper"] for t in a] == [t["input"]["paper"] for t in b]


def test_envelopes_have_required_fields_and_tags():
    tasks = list(generator.generate("v1", seed=0, split="revealed"))
    assert tasks
    for t in tasks:
        assert {"task_id", "arena_id", "task_set_version", "difficulty", "input"} <= t.keys()
        assert t["split"] == "revealed"
        assert t["visibility"] == "public"
        assert {"preregistration", "paper", "dimensions"} <= t["input"].keys()


def test_private_split_is_held_out():
    tasks = list(generator.generate("v1", seed=9, split="private"))
    assert all(t["split"] == "private" and t["visibility"] == "held_out" for t in tasks)


def test_all_six_tiers_present():
    tiers = {t["difficulty"]["tier"] for t in generator.generate("v1", seed=0)}
    assert tiers == {1, 2, 3, 4, 5, 6}


def test_t1_and_t2_have_no_deviations():
    for t in generator.generate("v1", seed=0):
        if t["difficulty"]["tier"] in (1, 2):
            assert t["difficulty"]["n_deviations"] == 0


def test_ground_truth_returns_dimension_records():
    tasks = list(generator.generate("v1", seed=0))
    gt = generator.ground_truth(tasks[0]["task_id"])
    assert "dimensions" in gt
    assert all({"dimension", "deviation", "deviation_kind"} <= d.keys() for d in gt["dimensions"])


def test_revealed_set_covers_every_deviation_kind():
    """The public benchmark must exercise the full array of injected deviations."""
    dims = generator._load_dimensions()
    all_kinds = {d["deviation_kind"] for d in dims}
    seen = set()
    for t in generator.generate("v1", seed=0, split="revealed"):
        gt = generator.ground_truth(t["task_id"])
        for d in gt["dimensions"]:
            if d["deviation"]:
                seen.add(d["deviation_kind"])
    assert seen == all_kinds


def test_splits_share_difficulty_matrix_but_differ_in_content():
    rev = list(generator.generate("v1", seed=0, split="revealed"))
    priv = list(generator.generate("v1", seed=12345, split="private"))

    def cells(tasks):
        return Counter(t["difficulty"]["tier"] for t in tasks)

    assert cells(rev) == cells(priv)
    assert [t["input"]["paper"] for t in rev] != [t["input"]["paper"] for t in priv]


# --- AbusingPreReg taxonomy extensions (modules 2, 5, 6, 9) -----------------

_NEW_KINDS = {
    "posthoc_results_leak",
    "unresolved_contingency",
    "nondirectional_as_confirmatory",
    "maineffect_to_interaction",
}


def test_task_count_matches_tier_formula():
    """n_tasks = 14 fixed (T1·4 + T2·4 + T5·4 + T6·2) + 2·n_dims (T3 + T4).

    Guards against arena.yaml#n_tasks drifting out of sync when dimensions change.
    """
    n_dims = len(generator._load_dimensions())
    tasks = list(generator.generate("v1", seed=0))
    assert len(tasks) == 14 + 2 * n_dims


def test_new_abusingprereg_kinds_present():
    kinds = {d["deviation_kind"] for d in generator._load_dimensions()}
    assert _NEW_KINDS <= kinds


def test_new_kinds_are_emitted_as_deviations():
    """Each new kind must actually appear as a flagged deviation in the revealed set."""
    seen = set()
    for t in generator.generate("v1", seed=0, split="revealed"):
        for d in generator.ground_truth(t["task_id"])["dimensions"]:
            if d["deviation"] and d["deviation_kind"] in _NEW_KINDS:
                seen.add(d["deviation_kind"])
    assert seen == _NEW_KINDS


def test_new_kind_modes_have_correct_deviation_gold():
    """Clean controls (consistent/paraphrase) are NOT deviations; subtle/deviated ARE.

    This is the G4 invariant at the gold level: a clean control that gold marks as
    a deviation would be a false-alarm trap the scorer could never reward.
    """
    import random as _random

    new_dims = [d for d in generator._load_dimensions() if d["deviation_kind"] in _NEW_KINDS]
    assert len(new_dims) == len(_NEW_KINDS)
    for dim in new_dims:
        for mode in ("consistent", "paraphrase"):
            _, _, gold = generator._render_dim(dim, _random.Random(1), mode)
            assert gold["deviation"] is False, f"{dim['deviation_kind']}/{mode} should be clean"
            assert gold["deviation_kind"] is None
        for mode in ("subtle", "deviated"):
            _, _, gold = generator._render_dim(dim, _random.Random(1), mode)
            assert gold["deviation"] is True, f"{dim['deviation_kind']}/{mode} should deviate"
            assert gold["deviation_kind"] == dim["deviation_kind"]


def test_new_kind_templates_render_without_keyerror():
    """Every mode template for the new dims must format with the available kwargs."""
    import random as _random

    new_dims = [d for d in generator._load_dimensions() if d["deviation_kind"] in _NEW_KINDS]
    for dim in new_dims:
        for mode in ("consistent", "paraphrase", "subtle", "deviated"):
            prereg_s, paper_s, _ = generator._render_dim(dim, _random.Random(7), mode)
            assert prereg_s and paper_s
            # no leftover unfilled placeholders
            assert "{planned}" not in paper_s and "{actual}" not in paper_s
            assert "{planned}" not in prereg_s and "{actual}" not in prereg_s
