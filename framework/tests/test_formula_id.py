"""Every arena must NAME the scoring formula its records were produced under.

WHY THIS IS A GATE AND NOT A STYLE RULE
---------------------------------------
`framework/runner.py::_build_instrument` stamps `instrument.formula_id` into
every run record. When an arena's scorer declares nothing it falls back to
``f"{arena_id}-unspecified"`` -- a deliberate refusal to invent a plausible name,
because "we do not know which formula this is" and "this is formula v1" are
different facts.

That fallback is honest and it is also a one-way door. Run records are immutable
history: once a tournament writes `<arena>-unspecified` into thousands of
records, the placeholder IS the published provenance for those scores, and the
only way to correct it is a rescore -- which is the exact operation the
`instrument` block was added to make distinguishable (2026-08-22, a published
mean moved 0.3980 -> 0.3887 under a changed formula with nothing in the record
to say so).

Measured 2026-08-27, before this test existed: **1 of 22 arenas declared a
FORMULA_ID** (stats-extraction-v1), and `scripts/verify_records.py` reported
`instrumented 0` -- no existing record carries an instrument block at all. So
this is purely forward-looking: nothing to migrate, and the entire cost of
getting it wrong falls on the NEXT tournament run. The newest run record was
2026-08-25, so that window was days wide, not weeks.

TWO WAYS TO DECLARE, BECAUSE ONE ARENA GENUINELY RUNS TWO FORMULAS
------------------------------------------------------------------
`pdf-text-fidelity-v1` dispatches on the SHAPE of the gold: a v1 task set
(`full_text`) is scored `0.5*Levenshtein + 0.5*token_F1`, a v2 task set
(`must_include`) by three-set scope scoring. A module constant would give both
the same name while they are incomparable -- the averaging error formula_id
exists to prevent. So a scorer may instead expose `formula_id_for(gt)` plus
`FORMULA_IDS`, the closed set of names it can emit.

WHAT EACH ASSERTION BUYS
------------------------
- declared: no arena silently rides the placeholder into new records.
- non-placeholder: a scorer cannot satisfy the gate by declaring the very string
  the fallback would have produced.
- unique: two arenas sharing an id would let a reader average incomparable rows.
- closed set: `formula_id_for` cannot return a name `FORMULA_IDS` does not list,
  so the uniqueness check above cannot be evaded by a dynamic name.
- CONTROLS: the runner's three outcomes (declared / nothing / hook failed) must
  stay three DISTINCT strings. Without these, deleting the fallback would turn
  every assertion above green while destroying the honest-refusal behaviour they
  are built on top of -- the trap increment 3 hit when its first probe measured
  its own fixture.
"""
from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from framework import runner

REPO = Path(__file__).resolve().parents[2]
ARENAS_DIR = REPO / "arenas"


def _arena_dirs() -> list[Path]:
    return sorted(d for d in ARENAS_DIR.iterdir() if (d / "scorer.py").is_file())


def _scorer(arena_dir: Path):
    return runner._import_arena_module(arena_dir, "scorer")


def _declared_ids(arena_dir: Path) -> list[str]:
    """Every formula name this arena can stamp, from either declaration style."""
    mod = _scorer(arena_dir)
    if callable(getattr(mod, "formula_id_for", None)):
        return list(getattr(mod, "FORMULA_IDS", ()) or ())
    fid = getattr(mod, "FORMULA_ID", None)
    return [fid] if isinstance(fid, str) and fid.strip() else []


#: Floor for the arena sweep. It must sit BELOW the public mirror's arena count,
#: not the full repo's: this file ships to the public package, where only the 16
#: seed-reproducible arenas exist (the other 6 author gold from real papers and
#: are never published -- see framework/publish.py). Asserting 22 here passed in
#: the private tree and would have failed on the mirror for a tree that was
#: perfectly healthy. Measured 2026-08-27: 22 private, 16 public.
_MIN_ARENAS = 10


def test_the_arena_set_is_non_empty():
    """A zero-length sweep would make every parametrised test below vacuous.

    The failure this guards is `ARENAS_DIR` resolving somewhere wrong, which
    yields 0 and turns every parametrised test into a silent pass.
    """
    found = _arena_dirs()
    assert len(found) >= _MIN_ARENAS, (
        f"found only {len(found)} arenas with a scorer under {ARENAS_DIR} "
        f"(expected >={_MIN_ARENAS}: 22 in the full repo, 16 in the public "
        f"package) -- the SWEEP, not the arenas, is what broke"
    )


@pytest.mark.parametrize("arena_dir", _arena_dirs(), ids=lambda d: d.name)
def test_every_arena_declares_a_formula_id(arena_dir: Path):
    ids = _declared_ids(arena_dir)
    assert ids, (
        f"{arena_dir.name}/scorer.py declares neither FORMULA_ID nor "
        f"(formula_id_for + FORMULA_IDS), so every record it scores is stamped "
        f"'{arena_dir.name}-unspecified' -- a placeholder that reads as "
        f"provenance and can only be corrected by a rescore"
    )
    for fid in ids:
        assert isinstance(fid, str) and fid.strip(), (
            f"{arena_dir.name} declares a non-string / empty formula id: {fid!r}"
        )


@pytest.mark.parametrize("arena_dir", _arena_dirs(), ids=lambda d: d.name)
def test_a_declared_formula_id_is_not_the_placeholder(arena_dir: Path):
    bad = {f"{arena_dir.name}-unspecified", f"{arena_dir.name}-formula-id-hook-failed"}
    for fid in _declared_ids(arena_dir):
        assert fid not in bad, (
            f"{arena_dir.name} declares a runner PLACEHOLDER ({fid!r}) as if it "
            f"were a real formula name; that satisfies the gate while asserting "
            f"nothing"
        )


def test_formula_ids_are_unique_across_arenas():
    seen: dict[str, str] = {}
    clashes: list[str] = []
    for d in _arena_dirs():
        for fid in _declared_ids(d):
            if fid in seen and seen[fid] != d.name:
                clashes.append(f"{fid}: {seen[fid]} and {d.name}")
            seen[fid] = d.name
    assert not clashes, (
        "two arenas share a formula_id, so their records look comparable and are "
        "not: " + "; ".join(clashes)
    )


# --------------------------------------------------------------------------
# The runner's three outcomes must stay three DISTINCT strings.
# --------------------------------------------------------------------------

def _mod(**attrs):
    m = types.ModuleType("_scorer_under_test")
    m.__file__ = None
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


def test_CONTROL_a_scorer_declaring_nothing_falls_back_to_unspecified():
    """Pins the honest-refusal branch, so the gate above cannot be satisfied by
    deleting the very behaviour it is built on top of."""
    assert runner._resolve_formula_id(_mod(), "some-arena-v1", {}) == \
        "some-arena-v1-unspecified"


def test_CONTROL_a_declared_constant_reaches_the_record():
    assert runner._resolve_formula_id(
        _mod(FORMULA_ID="some-arena-PxR-v1"), "some-arena-v1", {}) == "some-arena-PxR-v1"


def test_a_hook_is_consulted_and_wins_over_the_constant():
    """A two-formula arena must not be silently collapsed to its constant."""
    m = _mod(FORMULA_ID="constant-v1", formula_id_for=lambda gt: f"hook-{gt['shape']}")
    assert runner._resolve_formula_id(m, "some-arena-v1", {"shape": "a"}) == "hook-a"
    assert runner._resolve_formula_id(m, "some-arena-v1", {"shape": "b"}) == "hook-b"


def test_a_hook_returning_nothing_falls_through_to_the_constant():
    m = _mod(FORMULA_ID="constant-v1", formula_id_for=lambda gt: None)
    assert runner._resolve_formula_id(m, "some-arena-v1", {}) == "constant-v1"


def test_a_BROKEN_hook_is_distinguishable_from_an_undeclared_arena():
    """The load-bearing control: an unlabelled fallback is the defect this
    project has a standing rule against. A hook that raises must NOT produce the
    same string as an arena that simply never declared one, or the bug lives in
    published provenance with nothing to reveal it."""
    def _boom(gt):
        raise RuntimeError("scorer bug")

    broken = runner._resolve_formula_id(_mod(formula_id_for=_boom), "some-arena-v1", {})
    undeclared = runner._resolve_formula_id(_mod(), "some-arena-v1", {})
    assert broken == "some-arena-v1-formula-id-hook-failed"
    assert broken != undeclared


def test_a_broken_hook_does_not_raise_through_the_runner():
    """Provenance metadata failing is not a reason to score a real player 0."""
    def _boom(gt):
        raise RuntimeError("scorer bug")

    inst = runner._build_instrument(_mod(formula_id_for=_boom), "a-v1", {}, {})
    assert inst["formula_id"] == "a-v1-formula-id-hook-failed"


def test_a_hook_returning_a_non_string_is_treated_as_broken():
    m = _mod(formula_id_for=lambda gt: 42)
    assert runner._resolve_formula_id(m, "a-v1", {}) == "a-v1-formula-id-hook-failed"


# --------------------------------------------------------------------------
# The real two-formula arena, tested against REAL gold from both task sets.
# A hand-built fixture here would measure itself (increment 3 section 3).
# --------------------------------------------------------------------------

_PTF = ARENAS_DIR / "pdf-text-fidelity-v1"


def _first_real_gold(version: str) -> dict:
    path = _PTF / "task_sets" / version / "_ground_truth.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    first = data[next(iter(data))]
    return first["ground_truth"]


@pytest.mark.skipif(not (_PTF / "scorer.py").is_file()
                    or not (_PTF / "task_sets" / "v1" / "_ground_truth.json").is_file()
                    or not (_PTF / "task_sets" / "v2" / "_ground_truth.json").is_file(),
                    reason="pdf-text-fidelity-v1's scorer and both task sets must "
                           "be present; none of them ship in the public package")
def test_the_two_formula_arena_names_its_two_formulas_differently():
    """The whole reason the hook exists.

    v1 gold and v2 gold are scored by different formulas from one module. If they
    ever stamped the same name, a reader could average them into one leaderboard
    row -- incomparable numbers presented as comparable.
    """
    mod = _scorer(_PTF)
    v1 = mod.formula_id_for(_first_real_gold("v1"))
    v2 = mod.formula_id_for(_first_real_gold("v2"))

    assert v1 and v2, f"a real gold produced no formula id: v1={v1!r} v2={v2!r}"
    assert v1 != v2, (
        f"both task sets stamp {v1!r}, but v1 gold is scored "
        f"0.5*Levenshtein+0.5*token_F1 and v2 gold by three-set scope scoring"
    )
    assert v1 in mod.FORMULA_IDS and v2 in mod.FORMULA_IDS, (
        "formula_id_for returned a name outside the declared FORMULA_IDS, so the "
        "cross-arena uniqueness check above cannot see it"
    )


@pytest.mark.skipif(not (_PTF / "scorer.py").is_file(),
                    reason="pdf-text-fidelity-v1 is not in the public package "
                           "(it authors gold from real papers, so publish.py "
                           "vetoes it); nothing to dispatch on")
def test_the_two_formula_arena_refuses_to_name_a_mixed_shape_gold():
    """`score()` returns primary 0.0 for a gold whose profiles disagree in shape.

    No formula ran, so there is no honest formula name -- but the reason must be
    NAMED, not merged into the generic placeholder. `-unspecified` would make the
    audit print "this arena never declared a FORMULA_ID" at an arena that declares
    TWO, sending a reader to add a constant that already exists while the real
    cause (gold from two task sets in one record) appears nowhere.

    Not a defensive branch: `score()` guards this because it HAPPENED -- Codex and
    Sonnet 5 found it 2026-08-21 and it reproduced at 1.0 before the fix.
    """
    mod = _scorer(_PTF)
    mixed = {"gold_ascii_greek": {"must_include": ["x"]},
             "gold_preserve_greek": {"full_text": "x"}}
    assert mod.formula_id_for(mixed) is None
    got = runner._resolve_formula_id(mod, "pdf-text-fidelity-v1", mixed)
    assert got == "pdf-text-fidelity-v1-formula-not-applicable"
    assert got != "pdf-text-fidelity-v1-unspecified", (
        "a declared-but-inapplicable formula must stay distinguishable from an "
        "arena that never declared one"
    )


def test_the_four_placeholder_outcomes_are_all_distinct():
    """One string per fact. Any two colliding is the unlabelled-fallback defect."""
    def _boom(gt):
        raise RuntimeError("scorer bug")

    outcomes = {
        "undeclared": runner._resolve_formula_id(_mod(), "a-v1", {}),
        "not_applicable": runner._resolve_formula_id(
            _mod(formula_id_for=lambda gt: None), "a-v1", {}),
        "hook_failed": runner._resolve_formula_id(
            _mod(formula_id_for=_boom), "a-v1", {}),
        "declared": runner._resolve_formula_id(_mod(FORMULA_ID="a-REAL-v1"), "a-v1", {}),
    }
    assert len(set(outcomes.values())) == 4, (
        f"two outcomes share a string, so two different facts are "
        f"indistinguishable downstream: {outcomes}"
    )


def test_a_hook_that_declines_still_defers_to_a_declared_constant():
    """`-formula-not-applicable` is only correct when there is no default to fall
    back on. A module with both means the hook is an override, not the only word."""
    m = _mod(FORMULA_ID="constant-v1", formula_id_for=lambda gt: None)
    assert runner._resolve_formula_id(m, "a-v1", {}) == "constant-v1"


# --------------------------------------------------------------------------
# A name that ENCODES A WEIGHT must keep matching the weight in the code.
#
# Raised by all three consult seats on 2026-08-27 (sonnet as finding 1, sol and
# grok in their premise sections): `p-curve-ACC85-CAL15-v1` is accurate today,
# and nothing stopped someone editing CORRECT_WEIGHT from 0.85 to 0.80 while the
# name kept saying 85. The gate above would have stayed green and the id would
# have started lying -- a plausible wrong value in published provenance with no
# test failing, which is this project's founding failure mode.
#
# CONVENTION these names follow, enforced below:
#   * a two-digit run after an uppercase letter is a PERCENTAGE WEIGHT
#     (`ACC85` -> 0.85), and must appear as a real coefficient of `primary`
#   * `x` between terms means MULTIPLY (`F1xKINDxCAL`)
#   * a single digit is never a weight (`F1`, `MEAN3`, `SCOPE3SET`, `BESTOF2GOLD`)
# --------------------------------------------------------------------------

import ast
import re

_WEIGHT_IN_NAME = re.compile(r"(?<=[A-Z])(\d{2})(?!\d)")


def _primary_coefficients(arena_dir: Path) -> set[float]:
    """Every numeric coefficient that actually reaches `primary` in this scorer.

    Reads the AST rather than the text: module-level weight constants are
    resolved to their values, so `CORRECT_WEIGHT * correct` contributes 0.85.
    """
    tree = ast.parse((arena_dir / "scorer.py").read_text(encoding="utf-8"))
    consts: dict[str, float] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) \
           and isinstance(node.value.value, (int, float)):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    consts[t.id] = float(node.value.value)

    found: set[float] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(getattr(t, "id", "") in ("primary", "composite")
                   for t in node.targets):
            continue
        for sub in ast.walk(node.value):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, (int, float)):
                found.add(float(sub.value))
            elif isinstance(sub, ast.Name) and sub.id in consts:
                found.add(consts[sub.id])
    return found


@pytest.mark.parametrize("arena_dir", _arena_dirs(), ids=lambda d: d.name)
def test_a_weight_encoded_in_the_name_matches_the_code(arena_dir: Path):
    coeffs = _primary_coefficients(arena_dir)
    for fid in _declared_ids(arena_dir):
        for digits in _WEIGHT_IN_NAME.findall(fid):
            want = int(digits) / 100.0
            assert any(abs(want - c) < 1e-9 for c in coeffs), (
                f"{arena_dir.name} declares {fid!r}, whose '{digits}' claims a "
                f"weight of {want}, but the coefficients actually reaching "
                f"`primary` are {sorted(coeffs)}. Either the weight changed and "
                f"the name did not, or the name never described this formula."
            )


def test_CONTROL_the_weight_reader_actually_finds_weights():
    """A two-sided control on the reader itself.

    If `_primary_coefficients` silently returned an empty set -- a moved
    assignment, a renamed variable, an AST change -- the test above would pass
    for every arena while checking nothing, which looks exactly like success.
    So assert it finds the weights we know are there.
    """
    pc = _primary_coefficients(ARENAS_DIR / "p-curve-v1")
    assert 0.85 in pc and 0.15 in pc, (
        f"the coefficient reader found {sorted(pc)} for p-curve-v1, which "
        f"declares CORRECT_WEIGHT=0.85 / CALIB_WEIGHT=0.15 -- the READER is "
        f"broken, and every assertion above is vacuous"
    )
    n_with_weights = sum(1 for d in _arena_dirs()
                         if any(_WEIGHT_IN_NAME.findall(f) for f in _declared_ids(d)))
    # THE FLOOR IS NOT A CONSTANT, because this suite runs in TWO repositories.
    # Measured 2026-08-29: 7 of the 22 arenas here carry a weight-encoded formula
    # id, but 4 of those 7 are PRIVATE arenas, so the public mirror ships only 3.
    # A hardcoded 7 is therefore unreachable in the mirror and made the public
    # suite red -- which is the "public repo with a failing badge" failure
    # `framework/publish.py` already names. Found by
    # `scripts/sync_public_mirror.sh --verify-tests`; the local suite cannot see it.
    #
    # The floor is derived from what is actually present, so the private repo's
    # gate is NOT weakened: with all 22 arenas it still demands 7.
    n_arenas = len(list(_arena_dirs()))
    floor = 7 if n_arenas >= 22 else 3
    assert n_with_weights >= floor, (
        f"only {n_with_weights} of {n_arenas} arena names carry an encoded weight "
        f"(floor {floor}); the convention regex has stopped matching and the gate "
        f"is checking nothing"
    )
