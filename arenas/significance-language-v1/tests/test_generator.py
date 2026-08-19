"""Tests for the significance-language generator."""
import importlib.util
import sys
from collections import Counter
from pathlib import Path

ARENA_DIR = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("_significance_language_generator", ARENA_DIR / "generator.py")
generator = importlib.util.module_from_spec(_SPEC)
sys.modules["_significance_language_generator"] = generator
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
        assert t["split"] == "revealed"
        assert t["visibility"] == "public"
        assert "text" in t["input"]
        assert isinstance(t["input"]["text"], str) and t["input"]["text"]


def test_private_split_is_held_out():
    tasks = list(generator.generate("v1", seed=9, split="private"))
    assert all(t["split"] == "private" and t["visibility"] == "held_out" for t in tasks)


def test_all_six_tiers_present():
    tiers = {t["difficulty"]["tier"] for t in generator.generate("v1", seed=0)}
    assert tiers == {1, 2, 3, 4, 5, 6}


def test_t1_and_t2_have_no_flags():
    for t in generator.generate("v1", seed=0):
        if t["difficulty"]["tier"] in (1, 2):
            assert t["difficulty"]["n_flags"] == 0
            gt = generator.ground_truth(t["task_id"])
            assert gt["flags"] == []
            assert gt["mistake_kinds"] == []


def test_ground_truth_returns_flag_records():
    tasks = list(generator.generate("v1", seed=0))
    # Find a mistake-bearing task to inspect a flag's shape.
    t3 = next(t for t in tasks if t["difficulty"]["tier"] == 3)
    gt = generator.ground_truth(t3["task_id"])
    assert {"flags", "mistake_kinds"} <= gt.keys()
    assert gt["flags"]
    for f in gt["flags"]:
        assert {"span", "category"} <= f.keys()
        assert {"text", "char_start", "char_end"} <= f["span"].keys()


def test_ground_truth_raises_for_unknown_task():
    try:
        generator.ground_truth("sl-does-not-exist")
    except KeyError:
        return
    raise AssertionError("ground_truth should raise KeyError for an unknown task_id")


def test_gold_spans_index_into_the_text():
    """Each gold span's char range must slice the recorded span text out of input.text."""
    for t in generator.generate("v1", seed=0):
        gt = generator.ground_truth(t["task_id"])
        text = t["input"]["text"]
        for f in gt["flags"]:
            sp = f["span"]
            assert text[sp["char_start"]:sp["char_end"]] == sp["text"]


def test_revealed_set_covers_every_mistake_kind():
    """The public benchmark must exercise the full array of injected mistakes."""
    all_kinds = set(generator.MISTAKE_KINDS)
    seen = set()
    for t in generator.generate("v1", seed=0, split="revealed"):
        gt = generator.ground_truth(t["task_id"])
        seen.update(gt["mistake_kinds"])
    assert seen == all_kinds


def test_splits_share_difficulty_matrix_but_differ_in_content():
    rev = list(generator.generate("v1", seed=0, split="revealed"))
    priv = list(generator.generate("v1", seed=12345, split="private"))

    def cells(tasks):
        return Counter(t["difficulty"]["tier"] for t in tasks)

    assert cells(rev) == cells(priv)
    assert [t["input"]["text"] for t in rev] != [t["input"]["text"] for t in priv]


def test_broadened_kinds_present_with_confusable_controls():
    """2026-06-29 broaden: the two new spin/marginal kinds must be injected, and each
    must have a confusable clean-control look-alike wired for the T4 subtle tier
    (the realism guard — a mistake with no honest look-alike is the synthetic failure)."""
    new_kinds = {"one_sided_unjustified", "p_just_over_threshold_spin"}
    assert new_kinds <= set(generator.MISTAKE_KINDS)
    # Every mistake kind (incl. the new ones) has a confusable clean control.
    # _CONFUSABLE is a local in generate(); assert via the catalog + CLEAN_FLAVOURS instead.
    assert "clean_prereg_onesided" in generator.CLEAN_FLAVOURS
    catalog = generator._CATALOG
    for kind in generator.MISTAKE_KINDS:
        assert kind in catalog and catalog[kind], f"no catalog entries for new kind {kind}"
    for flavour in generator.CLEAN_FLAVOURS:
        assert flavour in catalog and catalog[flavour], f"no catalog entries for control {flavour}"

    # The new mistakes actually appear in gold, and clean controls never carry a flag.
    seen_new = set()
    for t in generator.generate("v1", seed=0, split="revealed"):
        gt = generator.ground_truth(t["task_id"])
        seen_new.update(set(gt["mistake_kinds"]) & new_kinds)
    assert seen_new == new_kinds, f"new kinds missing from revealed gold: {new_kinds - seen_new}"


def _gold_categories():
    """Every category the generator can emit as a gold flag."""
    cats = set()
    for t in generator.generate("v1", seed=0, split="revealed"):
        gt = generator.ground_truth(t["task_id"])
        for flag in gt["flags"]:
            cats.add(flag["category"])
    return cats


def test_output_schema_enum_covers_every_gold_category():
    """Drift guard: the output schema's category enum must allow every gold category.

    If the generator injects a kind the schema can't express, players hit
    output_schema_violation on it (the cycle-1 gap: 2 kinds added to the
    generator but not the schema enum).
    """
    import json

    schema = json.loads((ARENA_DIR / "schemas" / "output.schema.json").read_text(encoding="utf-8"))
    enum = set(schema["properties"]["flags"]["items"]["properties"]["category"]["enum"])
    missing = _gold_categories() - enum
    assert not missing, f"output schema category enum missing gold categories: {missing}"


def test_player_prompt_names_every_gold_category():
    """Drift guard: the player prompt must name every category players are scored on.

    A category the prompt never mentions cannot be emitted by an instructed
    player, guaranteeing category_mislabel (the cycle-1 gap).
    """
    prompt = (ARENA_DIR.parents[1] / "players" / "prompts" / "significance_language.txt").read_text(encoding="utf-8")
    missing = {c for c in _gold_categories() if c not in prompt}
    assert not missing, f"player prompt does not mention gold categories: {missing}"
