"""Tests for the grim-consistency generator."""
import importlib.util
import sys
from collections import Counter
from pathlib import Path

ARENA_DIR = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("_grim_consistency_generator", ARENA_DIR / "generator.py")
generator = importlib.util.module_from_spec(_SPEC)
sys.modules["_grim_consistency_generator"] = generator
_SPEC.loader.exec_module(generator)


def test_generate_is_deterministic():
    a = list(generator.generate("v1", seed=3))
    b = list(generator.generate("v1", seed=3))
    assert [t["task_id"] for t in a] == [t["task_id"] for t in b]
    assert [t["input"]["statistics"] for t in a] == [t["input"]["statistics"] for t in b]


def test_envelopes_have_required_fields_and_tags():
    tasks = list(generator.generate("v1", seed=0, split="revealed"))
    assert tasks
    for t in tasks:
        assert {"task_id", "arena_id", "task_set_version", "difficulty", "input"} <= t.keys()
        assert t["arena_id"] == "grim-consistency-v1"
        assert t["split"] == "revealed"
        assert t["visibility"] == "public"
        assert "statistics" in t["input"]
        for s in t["input"]["statistics"]:
            assert {"stat_id", "label", "stat_type", "n", "decimals"} <= s.keys()
            if s["stat_type"] == "mean":
                assert {"mean", "sd", "n_items", "scale_min", "scale_max"} <= s.keys()
            else:
                assert s["stat_type"] == "percent"
                assert "percent" in s


def test_private_split_is_held_out():
    tasks = list(generator.generate("v1", seed=9, split="private"))
    assert all(t["split"] == "private" and t["visibility"] == "held_out" for t in tasks)


def test_all_six_tiers_present():
    tiers = {t["difficulty"]["tier"] for t in generator.generate("v1", seed=0)}
    assert tiers == {1, 2, 3, 4, 5, 6}


def test_t1_and_t2_have_no_issues():
    for t in generator.generate("v1", seed=0):
        if t["difficulty"]["tier"] in (1, 2):
            assert t["difficulty"]["n_issues"] == 0
            gt = generator.ground_truth(t["task_id"])
            assert gt["mistake_kinds"] == []
            assert all(not g["flagged"] for g in gt["records"])


def test_ground_truth_returns_records_and_mistake_kinds():
    tasks = list(generator.generate("v1", seed=0))
    gt = generator.ground_truth(tasks[0]["task_id"])
    assert "records" in gt
    assert "mistake_kinds" in gt and isinstance(gt["mistake_kinds"], list)
    assert all({"stat_id", "issue_kind", "flagged"} <= g.keys() for g in gt["records"])


def test_ground_truth_missing_task_raises_keyerror():
    import pytest
    list(generator.generate("v1", seed=0))
    with pytest.raises(KeyError):
        generator.ground_truth("grim-t1-0-sDOES_NOT_EXIST")


def test_revealed_set_covers_every_issue_kind():
    """The public benchmark must exercise the full array of injected issues."""
    all_kinds = set(generator.ISSUE_KINDS)
    seen = set()
    for t in generator.generate("v1", seed=0, split="revealed"):
        gt = generator.ground_truth(t["task_id"])
        for g in gt["records"]:
            if g["flagged"]:
                seen.add(g["issue_kind"])
    assert seen == all_kinds


def test_splits_share_difficulty_matrix_but_differ_in_content():
    rev = list(generator.generate("v1", seed=0, split="revealed"))
    priv = list(generator.generate("v1", seed=12345, split="private"))

    def cells(tasks):
        return Counter(t["difficulty"]["tier"] for t in tasks)

    assert cells(rev) == cells(priv)
    assert [t["input"]["statistics"] for t in rev] != [t["input"]["statistics"] for t in priv]


def test_arena_is_grim_only():
    """This arena tests GRIM granularity only — of a mean, or of a percentage.

    GRIMMER / SD-granularity must NEVER appear here: it belongs to
    sprite-plausibility-v1, and a prior attempt to inject it shipped a gold bug
    (see the arena CHANGELOG "Fixed"). Guard the whole family, not just the one
    name that was reverted.
    """
    assert generator.ISSUE_KINDS == [
        "grim_inconsistent",
        "grim_percent_inconsistent",
    ]
    assert not any("grimmer" in k or "sd_" in k for k in generator.ISSUE_KINDS)


def test_mean_statistics_are_single_item():
    """GRIM-only convention: n_items must be 1 on every MEAN statistic so the mean
    granularity is 1/N, matching scrutiny::grim(). Percentage statistics carry no
    n_items (their granularity is 100/N)."""
    for split, seed in (("revealed", 0), ("private", 5)):
        for t in generator.generate("v1", seed=seed, split=split):
            for s in t["input"]["statistics"]:
                if s["stat_type"] == "mean":
                    assert s["n_items"] == 1, f"{s['stat_id']} has n_items != 1"
                else:
                    assert "n_items" not in s, f"{s['stat_id']} percent carries n_items"


def _grim_ok(s: dict) -> bool:
    """Independent GRIM check dispatching on the statistic's declared type."""
    if s["stat_type"] == "percent":
        return generator.grim_percent_consistent(s["percent"], s["n"], s["decimals"])
    return generator.grim_consistent(
        s["mean"], s["n"], s["n_items"], s["decimals"], s["scale_min"], s["scale_max"])


def test_gold_is_self_consistent():
    """Every gold-flagged value must FAIL an independent GRIM check, and every clean
    value must PASS it. This proves the gold was computed (not asserted): the arena
    never emits a false grim flag, and never calls a truly-impossible value clean.
    Covers both statistic types."""
    for split, seed in (("revealed", 0), ("private", 777)):
        tasks = list(generator.generate("v1", seed=seed, split=split))
        for t in tasks:
            gt = generator.ground_truth(t["task_id"])
            gold_by_id = {g["stat_id"]: g for g in gt["records"]}
            for s in t["input"]["statistics"]:
                g = gold_by_id[s["stat_id"]]
                ok = _grim_ok(s)
                if g["flagged"]:
                    assert not ok, f"{s['stat_id']} flagged grim but is GRIM-consistent"
                else:
                    assert ok, f"clean {s['stat_id']} fails GRIM"


def test_issue_kind_matches_statistic_type():
    """A percentage must be flagged grim_percent_inconsistent and a mean
    grim_inconsistent — never crossed. A crossed kind would be scored as a
    kind_mislabel against every correct player."""
    for split, seed in (("revealed", 0), ("private", 31)):
        for t in generator.generate("v1", seed=seed, split=split):
            gt = generator.ground_truth(t["task_id"])
            by_id = {s["stat_id"]: s for s in t["input"]["statistics"]}
            for g in gt["records"]:
                if not g["flagged"]:
                    continue
                expected = ("grim_percent_inconsistent"
                            if by_id[g["stat_id"]]["stat_type"] == "percent"
                            else "grim_inconsistent")
                assert g["issue_kind"] == expected, (
                    f"{g['stat_id']} is {by_id[g['stat_id']]['stat_type']} but gold "
                    f"kind is {g['issue_kind']}")


def test_percentages_are_actually_exercised():
    """Both statistic types must appear, and BOTH must carry injected issues.

    Guards the silent-half-fix failure mode: a percentage type that only ever
    appears as a clean control would let a player score 1.0 by never flagging a
    percentage at all.
    """
    seen_types, flagged_types = set(), set()
    for t in generator.generate("v1", seed=0, split="revealed"):
        gt = generator.ground_truth(t["task_id"])
        by_id = {s["stat_id"]: s for s in t["input"]["statistics"]}
        for s in t["input"]["statistics"]:
            seen_types.add(s["stat_type"])
        for g in gt["records"]:
            if g["flagged"]:
                flagged_types.add(by_id[g["stat_id"]]["stat_type"])
    assert seen_types == {"mean", "percent"}
    assert flagged_types == {"mean", "percent"}


def test_clean_percent_controls_exist_and_are_consistent():
    """The T2/T4 percentage traps: achievable-but-odd percentages that must NOT be
    flagged. Without these, 'flag every percentage' would be a winning strategy."""
    n_clean_percent = 0
    for t in generator.generate("v1", seed=0, split="revealed"):
        gt = generator.ground_truth(t["task_id"])
        gold_by_id = {g["stat_id"]: g for g in gt["records"]}
        for s in t["input"]["statistics"]:
            if s["stat_type"] == "percent" and not gold_by_id[s["stat_id"]]["flagged"]:
                n_clean_percent += 1
                assert generator.grim_percent_consistent(
                    s["percent"], s["n"], s["decimals"]), \
                    f"clean control {s['stat_id']} ({s['percent']}% of {s['n']}) is not achievable"
    assert n_clean_percent >= 6, f"only {n_clean_percent} clean percentage controls"


def test_grim_percent_consistent_matches_known_scrutiny_values():
    """Pin the percentage GRIM arithmetic to values verified against scrutiny 0.6.1
    (2026-08-04). At n=63: 27/63=42.857->'42.9' achievable, 28/63=44.444->'44.4'
    achievable, '43.0' achievable by no integer count."""
    assert generator.grim_percent_consistent(42.9, 63, 1) is True
    assert generator.grim_percent_consistent(44.4, 63, 1) is True
    assert generator.grim_percent_consistent(43.0, 63, 1) is False


def test_clean_stats_have_no_mistake_kinds_in_gold():
    for t in generator.generate("v1", seed=0):
        gt = generator.ground_truth(t["task_id"])
        flagged_kinds = {g["issue_kind"] for g in gt["records"] if g["flagged"]}
        assert flagged_kinds == set(gt["mistake_kinds"])


# --------------------------------------------------------------------------- #
# Taxonomy drift guards.
#
# An arena's mistake taxonomy lives in FOUR places that must stay in sync:
# the generator's ISSUE_KINDS, the gold `issue_kind` values, the output schema's
# enum, and the player PROMPT. Broadening any one without the others is a silent
# half-fix — the new kind becomes untestable (schema rejects it as an
# output_schema_violation, or the prompt never names it so the player is
# guaranteed a kind_mislabel) and scores look real but are systematically
# deflated. Standard since cycle 4 (2026-07-01); see _project/lessons.md.
# --------------------------------------------------------------------------- #
_REPO_ROOT = ARENA_DIR.parents[1]


def test_output_schema_enum_covers_every_gold_category():
    import json
    schema = json.loads(
        (ARENA_DIR / "schemas" / "output.schema.json").read_text(encoding="utf-8"))
    enum = schema["properties"]["records"]["items"]["properties"]["issue_kind"]["enum"]
    for kind in generator.ISSUE_KINDS:
        assert kind in enum, f"{kind} missing from output schema issue_kind enum"
    # And nothing stale lingers in the enum.
    assert {e for e in enum if e is not None} == set(generator.ISSUE_KINDS)


def test_player_prompt_names_every_gold_category():
    prompt = (_REPO_ROOT / "players" / "prompts" / "grim_consistency.txt").read_text(
        encoding="utf-8")
    for kind in generator.ISSUE_KINDS:
        assert kind in prompt, f"{kind} is never named in the player prompt"


def test_input_schema_declares_both_statistic_types():
    """Every field the generator emits must be declared: the input schema sets
    additionalProperties=false, so an undeclared field makes every task invalid."""
    import json
    schema = json.loads(
        (ARENA_DIR / "schemas" / "input.schema.json").read_text(encoding="utf-8"))
    props = schema["properties"]["statistics"]["items"]["properties"]
    emitted: set[str] = set()
    for t in generator.generate("v1", seed=0, split="revealed"):
        for s in t["input"]["statistics"]:
            emitted |= set(s.keys())
    missing = emitted - set(props)
    assert not missing, f"generator emits undeclared input fields: {sorted(missing)}"
    assert set(props["stat_type"]["enum"]) == {"mean", "percent"}


def test_reference_adapter_handles_every_statistic_type():
    """The scrutiny reference adapter is this arena's gold oracle (cycle-gate check
    c). If it cannot dispatch on a statistic type, it silently scores that type as
    unflagged and the 'gold cross-validates at 1.00' claim becomes false."""
    adapter = (_REPO_ROOT / "players" / "adapters" / "grim_scrutiny.R").read_text(
        encoding="utf-8")
    assert "percent = TRUE" in adapter, "adapter never runs the percentage GRIM test"
    for kind in generator.ISSUE_KINDS:
        assert kind in adapter, f"adapter can never emit {kind}"
