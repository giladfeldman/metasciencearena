"""Generator contract tests for code-translation-r-v1."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ARENA_DIR = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


generator = _load("_xlat_generator", ARENA_DIR / "generator.py")

REQUIRED_ENVELOPE_KEYS = {
    "task_id", "arena_id", "task_set_version", "split", "visibility",
    "difficulty", "input",
}


def test_generate_is_deterministic():
    a = [e["task_id"] for e in generator.generate("v1", 3)]
    b = [e["task_id"] for e in generator.generate("v1", 3)]
    assert a == b
    assert len(a) == len(set(a)), "task_ids must be unique"


def test_envelopes_carry_required_keys():
    for env in generator.generate("v1", 0):
        assert REQUIRED_ENVELOPE_KEYS <= set(env), f"missing keys in {env['task_id']}"
        assert env["arena_id"] == "code-translation-r-v1"


def test_revealed_is_public_private_is_held_out():
    assert all(e["visibility"] == "public" for e in generator.generate("v1", 0, "revealed"))
    assert all(e["visibility"] == "held_out" for e in generator.generate("v1", 1, "private"))


def test_covers_both_languages_at_every_tier():
    """The source_language axis must be crossed with tier, not confounded with it.

    If some analyses existed only in SPSS, a language comparison would really be
    measuring which analyses each language happened to get.
    """
    seen: dict[int, set[int]] = {}
    for env in generator.generate("v1", 0):
        d = env["difficulty"]
        seen.setdefault(d["tier"], set()).add(d["source_language"])

    # Derive the expected tiers from arena.yaml rather than hardcoding them:
    # a hardcoded {1..6} broke the moment tiers 7-9 were added (2026-08-04), and
    # a test that must be edited every time the ladder grows trains you to edit
    # it without thinking. The INVARIANT is "every declared tier is generated in
    # both languages", not "there are exactly six tiers".
    import yaml
    manifest = yaml.safe_load((ARENA_DIR / "arena.yaml").read_text(encoding="utf-8"))
    tier_axis = next(a for a in manifest["difficulty_axes"] if a["id"] == "tier")
    expected = set(range(tier_axis["min"], tier_axis["max"] + 1))

    assert set(seen) == expected, f"tiers present: {sorted(seen)}, declared: {sorted(expected)}"
    for tier, langs in seen.items():
        assert langs == {1, 2}, f"tier {tier} missing a language: {langs}"


def test_input_matches_schema_shape():
    for env in generator.generate("v1", 0):
        inp = env["input"]
        assert inp["source_language"] in {"spss", "stata"}
        assert inp["source_code"].strip(), "source_code must not be empty"
        assert inp["dataset_path"].endswith(".csv")
        assert inp["required_statistics"], "every task needs required_statistics"
        assert inp["codebook"], "codebook must be provided"


def test_codebook_declares_the_user_missing_code():
    """The 99 trap is fair only because the codebook DECLARES the code.

    Otherwise the task would test guessing, not translation.
    """
    env = next(iter(generator.generate("v1", 0)))
    item4 = [c for c in env["input"]["codebook"] if c["name"] == "item4"]
    assert item4 and item4[0].get("missing_code") == 99


def test_ground_truth_round_trips():
    for env in generator.generate("v1", 0):
        gt = generator.ground_truth(env["task_id"])
        assert gt["required_statistics"] == env["input"]["required_statistics"]
        assert gt["tier"] == env["difficulty"]["tier"]


def test_mistake_kinds_present_for_trap_tiers():
    """Parity matches on mistake_kinds, so trap tiers must expose labels."""
    by_tier: dict[int, list[str]] = {}
    for env in generator.generate("v1", 0):
        gt = generator.ground_truth(env["task_id"])
        by_tier[gt["tier"]] = gt["mistake_kinds"]
    assert by_tier[1] == [], "tier 1 is the clean control — no traps"
    for tier in (2, 3, 4, 5, 6):
        assert by_tier[tier], f"tier {tier} must declare its trap kinds"


def test_dataset_exists_and_has_trap_features():
    csv = generator.dataset_csv("wellbeing")
    assert csv.exists(), "run tools/make_dataset.py"
    text = csv.read_text(encoding="utf-8").splitlines()
    header = text[0].split(",")
    assert "item4" in header and "group" in header and "condition" in header
    rows = [r.split(",") for r in text[1:] if r.strip()]
    assert len(rows) == 180
    i4 = header.index("item4")
    assert any(r[i4] == "99" for r in rows), "need the 99 user-missing code"
    age = header.index("age")
    assert any(r[age] == "" for r in rows), "need genuine NAs for the deletion trap"


@pytest.mark.parametrize("analysis", [
    "descriptives", "ttest_groups", "regression_multi",
    "recode_transform", "anova_factorial", "pipeline_select_model",
])
def test_gold_is_built_for_every_analysis(analysis):
    """Gold must exist and cover every declared statistic.

    Missing gold makes the scorer return `gold_not_built` for that task, which
    would silently drop it from the leaderboard rather than fail loudly.
    """
    gold = generator.load_gold(analysis)
    assert gold is not None, f"no gold for {analysis} — run tools/build_gold.py"
    entry = next(e for e in generator._catalog() if e["id"] == analysis)
    assert set(entry["gold_statistics"]) <= set(gold)
