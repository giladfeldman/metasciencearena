"""Task set v4 pins the defect it was minted to fix (2026-08-28).

v3 was MANUFACTURING the reporting errors it graded. `generator._round_p` clamps
any true p below .0005 to print `p = .001` and any above .999 to `p = .999`;
both are false statements about the statistic, so a tool that recomputes p is
right to flag them. Measured on v3 revealed: 44 of the 45 truthful NHST items
labelled incoherent sit at a clamp, and exactly ONE is a genuine mismatch.

Every test here states the MEASURED v3 value it is moving away from, and every
one was watched fail against v3 before v4 existed -- `test_..._is_a_v4_property`
is the two-sided control that keeps that honest, by asserting the v3 value is
still the OLD one. If someone "fixes" v4 by changing v3, that control goes red.

Full write-up: the 2026-08-27 stats-arena correction and agy-containment notes.
"""
from pathlib import Path
import importlib.util
import sys

ARENA_DIR = Path(__file__).resolve().parents[1]


def _load(mod_name, filename):
    spec = importlib.util.spec_from_file_location(mod_name, ARENA_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


generator = _load("_stats_extraction_v1_generator_v4", "generator.py")
scorer = _load("_stats_extraction_v1_scorer_v4", "scorer.py")


def _truthful_nhst(version):
    for t in generator.generate(version, 0, "revealed"):
        for it in generator.ground_truth(t["task_id"])["items"]:
            if it["kind"] == "nhst_stat" and it.get("truthful", True):
                yield it


def _n_incoherent(version):
    return sum(1 for it in _truthful_nhst(version)
               if it.get("expected_verdict") and it.get("coherent") is False)


# --- the defect: clamped p values printed as point values --------------------

def test_no_truthful_item_is_incoherent():
    """v3 measured 45, of which 44 were our own printing clamps. A truthful item
    whose reported p is not true of its statistic is a defect WE manufactured."""
    bad = [(it["fields"].get("p"), it["fields"].get("p_rel"), it.get("p_recomputed"))
           for it in _truthful_nhst("v4")
           if it.get("expected_verdict") and it.get("coherent") is False]
    assert not bad, f"{len(bad)} truthful NHST items are incoherent on v4: {bad[:5]}"


def test_the_incoherent_class_is_a_v4_property_not_a_relabelled_v3():
    """TWO-SIDED CONTROL. v3 must STILL measure 45; if it does not, someone made
    v4 green by editing frozen history instead of by fixing the printing."""
    assert _n_incoherent("v3") == 45, (
        f"v3 now measures {_n_incoherent('v3')} incoherent truthful items, not 45 "
        "- frozen history appears to have been rewritten")


def test_the_inequality_is_emitted_and_is_true_of_the_statistic():
    """The point of v4: `p < .001` / `p > .999` where the true p really is beyond
    the grid. v3 emitted the inequality ZERO times."""
    rels = {"<": 0, ">": 0, "=": 0}
    for it in _truthful_nhst("v4"):
        f = it["fields"]
        rel = f.get("p_rel")
        if rel is None:
            continue
        rels[rel] += 1
        tp = generator._two_sided_p(f["test_type"], f["value"], f["df1"], f["df2"])
        if tp is None:
            continue
        if rel == "<":
            assert tp < f["p"], f"printed p < {f['p']} but true p is {tp}"
        elif rel == ">":
            assert tp > f["p"], f"printed p > {f['p']} but true p is {tp}"
    assert rels["<"] > 0 and rels[">"] > 0, f"v4 emits no inequality at all: {rels}"
    assert rels["="] > 0, (
        f"EVERY p became an inequality: {rels}. A p that genuinely rounds to .001 "
        "must keep its point value - converting blindly made 11 correct items wrong.")


def test_a_p_that_genuinely_rounds_to_the_floor_keeps_its_point_value():
    """The trap that made the first implementation wrong. `p < .001` is FALSE of a
    statistic whose true p is .0012, so the relation must key on the TRUE p."""
    # true p comfortably inside [.0005, .0015) -> must stay "="
    assert generator._apa_p_rel("t", 3.5, 12, 0, 0.001) == "="
    # true p far below the floor -> "<"
    assert generator._apa_p_rel("t", 12.0, 100, 0, 0.001) == "<"
    # no closed form -> unchanged
    assert generator._apa_p_rel("U", 5.0, 10, 0, 0.03) == "="


# --- the consequence: the forgiveness rule has no cases ----------------------

def test_the_forgiveness_cell_has_no_cases_on_v4():
    """v3 measured 45 forgiven items, which is why statcheck was graded on 39 of
    73 while escimate was graded on 70 of 72. The scorer is UNCHANGED; the rule
    is inert because the arena no longer manufactures incoherent items."""
    forgiven = 0
    for t in generator.generate("v4", 0, "revealed"):
        for it in generator.ground_truth(t["task_id"])["items"]:
            checkable, _expected, coherent = scorer._gold_verdict(it)
            if checkable and not coherent:
                forgiven += 1
    assert forgiven == 0, f"{forgiven} v4 items still reach the forgiveness cell"


def test_the_forgiveness_cell_still_fires_when_it_should():
    """CONTROL for the test above: a zero is only meaningful if the cell works.
    Without this, deleting the cell entirely would also make that test pass."""
    assert scorer._verification_cell("ok", "inconsistent", False) == "abstain"
    assert scorer._verification_cell("ok", "inconsistent", True) == "false_alarm"


# --- what v4 must NOT have changed -------------------------------------------

def test_the_deliberate_deception_class_is_intact():
    """v4 removes the ACCIDENTAL error class. Removing the deliberate one too
    would leave an arena that cannot distinguish any player from any other."""
    kinds = {}
    for t in generator.generate("v4", 0, "revealed"):
        for it in generator.ground_truth(t["task_id"])["items"]:
            if not it.get("truthful", True):
                kinds[it.get("deception_kind")] = kinds.get(it.get("deception_kind"), 0) + 1
    assert sum(kinds.values()) == 24, f"deception count moved: {kinds}"
    assert len(kinds) == 8, f"a deception KIND disappeared: {sorted(kinds)}"


def test_the_clean_class_is_not_empty():
    """Same floor as v3's guard. Measured on v4: 89. The fold on v3 alone gave 36,
    which is why that option was measured and rejected rather than shipped."""
    n_ok = sum(1 for it in _truthful_nhst("v4") if it.get("expected_verdict") == "ok")
    assert n_ok >= 40, f"only {n_ok} genuinely coherent truthful NHST items"
