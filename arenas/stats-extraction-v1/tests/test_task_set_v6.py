"""v6 makes gold COMPOSITION seed-independent, so the two splits match (2026-09-22).

Through v5, `generator._build_task` chose each density slot's clause shape from
`rng`: the kind (`rng.choice(["nhst", "effect_size"])`) and, from v3, whether an
nhst slot also printed its effect size (`rng.random() < 0.5`). The revealed and
private splits draw from DIFFERENT seeds, so the two halves of the benchmark
ended up made of different things.

WHY THAT IS A SCORING DEFECT AND NOT BOOKKEEPING. `scorer.py`'s `fn` term charges
every unmatched gold item as a false negative, so a tool that only reads
significance tests -- statcheck, the instrument this arena is built around -- has
its per-task recall capped at nhst_items / all_items. Through v5 that cap is a
property of the SEED, not of the tool. MEASURED on v5 at the real split seeds,
mean over all 36 tasks:

    NHST-only recall ceiling   revealed 0.6516   private 0.6901   gap -0.0385

    per-tier nhst/all:  t1 11/28 vs 11/26   t2 8/12 vs 7/12   t3 5/12 vs 7/12
                        t4 18/18 both       t5 9/12 both      t6 53/81 vs 51/81

Tiers 2 and 3 carry IDENTICAL item counts on both splits and still differ in
composition. That is why pinning cardinality alone -- the `rng.random() < 0.5`
draw, the one that is easy to spot -- would have left the larger half standing.

Every test here has a TWO-SIDED CONTROL asserting v5 still measures its old
seed-dependent value. A generator that returned nothing, or that cloned the two
splits, would make the v6 assertions trivially true; the v5 controls turn both of
those into a red test instead of a green one.

THE REJECTED OPTION, recorded because it is the tempting one: restart the index
cycle at every task. That is seed-independent too, and wrong -- a task holds only
1-3 slots, so a short cycle over-weights its first entry. Measured, that variant
put 117 nhst items in 168 where v5 averages 104.95 in 162.05, silently making v6
an easier arena than the one it replaces. `generate` therefore carries a running
per-tier slot counter across tasks (`v6_slot_offsets`), the same shape as tier
5's `t5_offset`, and the composition lands on v5's mean instead.
"""
from pathlib import Path
import importlib.util
import statistics
import sys

import pytest

ARENA_DIR = Path(__file__).resolve().parents[1]

#: The task set under test, and the ones it must not have disturbed.
CURRENT = "v6"
FROZEN = ("v1", "v2", "v3", "v4", "v5")

#: Seed budget. 40 is the count the arena's other seed-sweep evidence uses
#: (arena.yaml#benchmark_splits.parity), so the numbers here are comparable.
N_SEEDS = 40

#: Revealed/private seed PAIRS for the cross-split test. Deliberately arbitrary:
#: the real private seed is gitignored, and the claim under test is about every
#: seed pair rather than about the two the arena happens to hold.
SEED_PAIRS = ((0, 101), (1, 202), (2, 303), (3, 404),
              (4, 505), (5, 606), (6, 707), (7, 808))


def _generator():
    spec = importlib.util.spec_from_file_location(
        "stats_extraction_generator_v6_test", ARENA_DIR / "generator.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gen():
    return _generator()


def _profile(gen, version, seed, split="revealed"):
    """One split's composition: per task, (tier, density, nhst items, all items).

    Deliberately excludes the task TEXT, so this is a fingerprint of what the
    tasks are made of and not of what they say.
    """
    out = []
    for env in gen.generate(version, seed, split=split):
        items = gen.ground_truth(env["task_id"])["items"]
        out.append((env["difficulty"]["tier"], env["difficulty"]["density"],
                    sum(1 for it in items if it.get("kind") == "nhst_stat"),
                    len(items)))
    # Vacuity guard: a generator that yielded nothing would make every assertion
    # in this file pass. 36 is the tier x density x n_per_cell matrix.
    assert len(out) == 36, f"{version} seed {seed} {split}: {len(out)} tasks, expected 36"
    assert all(a > 0 for _, _, _, a in out), "a task carries no gold items at all"
    return tuple(out)


def _profiles(gen, version):
    return {_profile(gen, version, s) for s in range(N_SEEDS)}


# --- the defect, and the control that keeps it honest ------------------------

def test_v6_composition_is_identical_on_every_seed(gen):
    """The fix: one composition profile across all 40 seeds."""
    profiles = _profiles(gen, CURRENT)
    assert len(profiles) == 1, (
        f"v6 composition still varies by seed: {len(profiles)} distinct profiles "
        f"over {N_SEEDS} seeds. Some draw in _build_task is still reading rng.")


def test_v5_composition_still_varies_by_seed(gen):
    """TWO-SIDED CONTROL for the test above.

    v5 is frozen history and must keep measuring its defect. If this goes green
    at 1 profile, v6's logic has leaked into a frozen version and the test above
    is passing for the wrong reason.
    """
    profiles = _profiles(gen, "v5")
    assert len(profiles) > 1, (
        "v5 composition is now seed-independent -- the v6 change has reached a "
        "frozen task set, which orphans its committed gold and published scores.")


def test_v6_splits_are_composition_matched(gen):
    """The point of the whole change: revealed and private are made of the same
    parts, whatever seeds they are drawn from."""
    mismatched = [(a, b) for a, b in SEED_PAIRS
                  if _profile(gen, CURRENT, a, "revealed")
                  != _profile(gen, CURRENT, b, "private")]
    assert not mismatched, f"v6 splits differ in composition for seed pairs {mismatched}"


def test_v5_splits_are_not_composition_matched(gen):
    """TWO-SIDED CONTROL: measured 0 of 8 pairs match on v5. If this ever passes,
    `_profile` has stopped discriminating and the test above proves nothing."""
    matched = [(a, b) for a, b in SEED_PAIRS
               if _profile(gen, "v5", a, "revealed")
               == _profile(gen, "v5", b, "private")]
    assert not matched, f"v5 splits unexpectedly match on {matched}; _profile is too coarse"


# --- v6 must not achieve parity by cloning the splits ------------------------

def test_v6_content_still_varies_by_seed(gen):
    """The cheapest way to pass every test above is to stop varying with the seed
    at all -- which would hand the private split's answers to anyone holding the
    revealed one. Composition is pinned; CONTENT must not be."""
    texts: dict[str, set[str]] = {}
    for seed in range(5):
        for env in gen.generate(CURRENT, seed, split="revealed"):
            cell = env["task_id"].rsplit("-s", 1)[0]
            texts.setdefault(cell, set()).add(env["input"]["text"])
    assert len(texts) == 36, f"{len(texts)} task cells, expected 36"
    collapsed = {c: len(v) for c, v in texts.items() if len(v) != 5}
    assert not collapsed, (
        f"v6 renders identical text across seeds for {collapsed} -- the splits are "
        "being cloned rather than matched.")


# --- v6 must not silently move the arena's difficulty ------------------------

def test_v6_lands_on_v5_mean_composition(gen):
    """v6 removes the variance without shifting the centre.

    MEASURED v5 over 40 seeds: 104.95 nhst items, 162.05 gold items, NHST-only
    ceiling 0.6545. v6 on every seed: 105, 162, 0.6508. The tolerances below are
    deliberately tight -- their job is to go red if someone changes a cycle in
    `_V6_SLOT_CYCLES` and moves the arena, not to accommodate that.
    """
    profile = next(iter(_profiles(gen, CURRENT)))
    n_nhst = sum(n for _, _, n, _ in profile)
    n_all = sum(a for _, _, _, a in profile)
    ceiling = statistics.fmean(n / a for _, _, n, a in profile)

    assert n_nhst == 105, f"v6 nhst items {n_nhst}, expected 105 (v5 mean 104.95)"
    assert n_all == 162, f"v6 gold items {n_all}, expected 162 (v5 mean 162.05)"
    assert ceiling == pytest.approx(0.6508, abs=0.005), (
        f"v6 NHST-only recall ceiling {ceiling:.4f}, expected ~0.6508 "
        "(v5 mean 0.6545). The arena's difficulty has moved.")


# --- the slot plan itself ----------------------------------------------------

def test_slot_plan_is_a_pure_function_of_the_index(gen):
    """No rng anywhere near it, and it cycles."""
    templates = {"effect_size": ["x"], "nhst_with_es": ["y"]}
    first = [gen._slot_plan_v6(i, templates) for i in range(12)]
    again = [gen._slot_plan_v6(i, templates) for i in range(12)]
    assert first == again
    assert first[:4] == first[4:8] == first[8:12], f"cycle does not repeat: {first}"


@pytest.mark.parametrize("has_es,has_nhst_es,expected", [
    (True, True, ("nhst_with_es", "effect_size", "nhst", "effect_size")),
    (True, False, ("nhst", "effect_size")),
    # The two below are FALLBACKS that no tier ships today. They are asserted
    # here precisely because nothing else exercises them: a future tier that
    # omits a template pool must not silently fall back to a seeded draw.
    (False, True, ("nhst_with_es", "nhst")),
    (False, False, ("nhst",)),
])
def test_every_template_shape_has_a_deterministic_cycle(gen, has_es, has_nhst_es, expected):
    templates = {}
    if has_es:
        templates["effect_size"] = ["x"]
    if has_nhst_es:
        templates["nhst_with_es"] = ["y"]
    got = tuple(gen._slot_plan_v6(i, templates) for i in range(len(expected)))
    assert got == expected


def test_v6_gate_covers_exactly_the_unfrozen_versions(gen):
    for v in FROZEN:
        assert not gen._is_v6_plus(v), f"{v} is frozen but takes the v6 code path"
    for v in ("v6", "v7", "v10"):
        assert gen._is_v6_plus(v), f"{v} should take the v6 code path"
