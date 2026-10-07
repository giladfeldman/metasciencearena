"""A declared step-aside must NOT be able to print as a full pass.

WHY THIS FILE EXISTS. stats-extraction-v1's revealed and private splits are not
matched on item-KIND composition: the v1-v5 branch of generator.py `_build_task` draws each
clause's kind from the seed and adds a second item on a coin flip (`with_es`). MEASURED on v5, an NHST-only
extractor's recall ceiling is 0.6516 revealed against 0.6901 private -- the two
halves are different instruments for statcheck.

The FIRST attempt at this fix compared the synthesised `clean` residual as a base
rate under a `clean_rate_tolerance`. Two consult seats and a reproduction killed it:
the v5 asymmetry lives in tier 1, where the clean rate is 1.0000 on BOTH splits, so
no tolerance at any value -- including 0 -- could ever see it. The gate printed
PARITY OK on a pair that is measurably non-comparable. That is the exact false green
the portfolio's "no pretending" rule exists to prevent, and these tests are the
guard against re-introducing it in any form.

The replacement declares the gap instead of gating it, and the declaration is scoped
to an EXPLICIT version list so it expires by construction.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import yaml

from framework import parity

REPO_ROOT = Path(__file__).resolve().parents[2]
STATS_ARENA = REPO_ROOT / "arenas" / "stats-extraction-v1"

_EXCUSED = (
    "    match_axes: [tier]\n"
    "    match_categories: true\n"
    "    count_tolerance: 0.0\n"
    "    composition_unmatched:\n"
    "      versions: [v3, v4, v5]\n"
    "      reason: measured ceiling gap 0.0385\n"
)
_STRICT = (
    "    match_axes: [tier]\n"
    "    match_categories: true\n"
    "    count_tolerance: 0.0\n"
)


def _fake_arena(tmp_path: Path, parity_block: str) -> Path:
    d = tmp_path / "fake-arena"
    d.mkdir()
    (d / "arena.yaml").write_text(
        "arena_id: fake-arena\n"
        "benchmark_splits:\n"
        "  revealed: {seed: 0}\n"
        "  private: {seed_source: env}\n"
        "  parity:\n" + parity_block,
        encoding="utf-8",
    )
    (d / "generator.py").write_text(
        "def generate(v, s, split='revealed'):\n    return []\n", encoding="utf-8")
    return d


def _coverage(rev_cats, priv_cats, rev_kinds=None, priv_kinds=None):
    """Patch _collect_coverage with fixed counts. Difficulty cells always match."""
    def fake(gen_mod, version, seed, split, axes):
        cells = Counter({(1,): 2})
        if split == "revealed":
            return cells, Counter(rev_cats), rev_kinds or {}
        return cells, Counter(priv_cats), priv_kinds or {}
    return fake


# --------------------------------------------------------------------------
# The headline must tell the truth.
# --------------------------------------------------------------------------

def test_an_excused_version_reports_PARTIAL_not_OK(tmp_path, monkeypatch):
    monkeypatch.setenv("SCIENCEARENA_PRIVATE_SEED", "999")
    monkeypatch.setattr(parity, "_collect_coverage",
                        _coverage({"clean": 139, "wrong_df": 24},
                                  {"clean": 137, "wrong_df": 24}))
    r = parity.check_parity(_fake_arena(tmp_path, _EXCUSED), "v5")
    assert r.ok, r.summary()
    assert r.not_enforced, "an excused run must SAY what it did not enforce"
    assert "PARITY PARTIAL" in r.summary()
    assert "PARITY OK" not in r.summary()


def test_a_clean_run_still_reports_OK(tmp_path, monkeypatch):
    """CONTROL -- PARTIAL must not become the universal answer."""
    monkeypatch.setenv("SCIENCEARENA_PRIVATE_SEED", "999")
    monkeypatch.setattr(parity, "_collect_coverage",
                        _coverage({"clean": 139, "wrong_df": 24},
                                  {"clean": 139, "wrong_df": 24}))
    r = parity.check_parity(_fake_arena(tmp_path, _STRICT), "v5")
    assert r.ok and not r.not_enforced
    assert "PARITY OK" in r.summary()


def test_an_arena_with_NO_declaration_keeps_the_old_strict_residual_check(
        tmp_path, monkeypatch):
    """The other 20 arenas must be completely unaffected by this change.

    This is the blast-radius test. `composition_unmatched` is opt-in, so an arena
    that declares nothing has to keep failing on a `clean` residual mismatch at
    `count_tolerance` exactly as it did before -- otherwise a fix scoped to one
    arena would have quietly loosened the gate for every arena in the fleet.
    """
    monkeypatch.setenv("SCIENCEARENA_PRIVATE_SEED", "999")
    monkeypatch.setattr(parity, "_collect_coverage",
                        _coverage({"clean": 139, "wrong_df": 24},
                                  {"clean": 137, "wrong_df": 24}))
    r = parity.check_parity(_fake_arena(tmp_path, _STRICT), "v5")
    assert not r.ok, "an undeclared arena must still fail on the residual"
    assert not r.not_enforced, "nothing was excused, so nothing may be reported as such"
    assert any("clean" in p and "count_tolerance" in p for p in r.problems), r.summary()


# --------------------------------------------------------------------------
# THE ACCEPTANCE TEST FOR v6: a version not on the list gets the strict check.
# --------------------------------------------------------------------------

def test_a_version_NOT_in_the_list_is_checked_STRICTLY(tmp_path, monkeypatch):
    """This is how the declaration expires, and it is the gate on the v6 fix.

    The generator fix lands behind a `v6` flag. v6 is deliberately absent from
    `composition_unmatched.versions`, so the moment it exists this path applies
    the full exact-count rule to it -- with nobody editing arena.yaml and nobody
    having to REMEMBER that the defect was fixed.
    """
    monkeypatch.setenv("SCIENCEARENA_PRIVATE_SEED", "999")
    monkeypatch.setattr(parity, "_collect_coverage",
                        _coverage({"clean": 139, "wrong_df": 24},
                                  {"clean": 137, "wrong_df": 24}))
    r = parity.check_parity(_fake_arena(tmp_path, _EXCUSED), "v6")
    assert not r.ok, "v6 is not excused, so the residual must be gated: " + r.summary()
    assert not r.not_enforced
    assert any("clean" in p for p in r.problems)


# --------------------------------------------------------------------------
# The anti-false-green tests: a step-aside must not loosen anything else.
# --------------------------------------------------------------------------

def test_an_excused_version_still_fails_a_declared_kind_mismatch(tmp_path, monkeypatch):
    """The excuse covers COMPOSITION only. The eight declared kinds stay exact."""
    monkeypatch.setenv("SCIENCEARENA_PRIVATE_SEED", "999")
    monkeypatch.setattr(parity, "_collect_coverage",
                        _coverage({"clean": 139, "wrong_df": 5},
                                  {"clean": 139, "wrong_df": 4}))
    r = parity.check_parity(_fake_arena(tmp_path, _EXCUSED), "v5")
    assert not r.ok
    assert any("wrong_df" in p for p in r.problems), r.summary()


def test_an_excused_version_still_fails_a_difficulty_cell_mismatch(tmp_path, monkeypatch):
    """Per-cell TASK counts are exact on every seed and stay gated."""
    monkeypatch.setenv("SCIENCEARENA_PRIVATE_SEED", "999")

    def fake(gen_mod, version, seed, split, axes):
        cats = Counter({"clean": 10})
        if split == "revealed":
            return Counter({(1,): 2, (2,): 2}), cats, {}
        return Counter({(1,): 2}), cats, {}
    monkeypatch.setattr(parity, "_collect_coverage", fake)
    r = parity.check_parity(_fake_arena(tmp_path, _EXCUSED), "v5")
    assert not r.ok
    assert any("difficulty cells" in p for p in r.problems), r.summary()


# --------------------------------------------------------------------------
# A declaration that excuses nothing must say so.
# --------------------------------------------------------------------------

def test_a_vacuous_declaration_announces_itself(tmp_path, monkeypatch):
    """A declaration covering no real difference trains readers to skim past it."""
    monkeypatch.setenv("SCIENCEARENA_PRIVATE_SEED", "999")
    same = {(1,): Counter({"nhst_stat": 3})}
    monkeypatch.setattr(parity, "_collect_coverage",
                        _coverage({"clean": 139}, {"clean": 139}, same, dict(same)))
    r = parity.check_parity(_fake_arena(tmp_path, _EXCUSED), "v5")
    assert any("VACUOUS" in x for x in r.not_enforced), r.summary()


# --------------------------------------------------------------------------
# The real arena.
# --------------------------------------------------------------------------

def test_the_stats_arena_declares_the_versions_explicitly():
    cfg = yaml.safe_load((STATS_ARENA / "arena.yaml").read_text(encoding="utf-8"))
    decl = cfg["benchmark_splits"]["parity"]["composition_unmatched"]
    assert decl["versions"] == ["v3", "v4", "v5"]
    assert "v6" not in decl["versions"], (
        "v6 must NOT be excused -- it is the version that fixes the generator, and "
        "its absence here is what makes the declaration expire on its own."
    )
    assert "0.6516" in decl["reason"] and "0.6901" in decl["reason"], (
        "the reason must carry the measurement, or it is an assertion"
    )


def test_the_old_false_green_tolerance_is_gone():
    """`clean_rate_tolerance` must never come back, in the arena OR the schema.

    It could not detect the v5 asymmetry at ANY value, because that asymmetry sits
    in tier 1 where the clean rate is 1.0000 on both splits. Re-introducing it
    would restore a gate that prints OK on a non-comparable pair.
    """
    arena = (STATS_ARENA / "arena.yaml").read_text(encoding="utf-8")
    schema = (REPO_ROOT / "framework" / "contract" / "schemas"
              / "arena_manifest.schema.json").read_text(encoding="utf-8")
    assert "clean_rate_tolerance: 0" not in arena
    assert '"clean_rate_tolerance"' not in schema


def test_the_real_arena_reports_PARTIAL_on_the_served_version():
    """End-to-end, against the real generator and the real seeds."""
    from framework.discovery import resolve_task_set_version
    manifest = yaml.safe_load((STATS_ARENA / "arena.yaml").read_text(encoding="utf-8"))
    r = parity.check_parity(STATS_ARENA, resolve_task_set_version(manifest))
    assert r.ok, r.summary()
    assert r.not_enforced, "v5 IS unmatched on composition; saying nothing is the bug"
    assert "PARITY PARTIAL" in r.summary()
    assert any("nhst_stat" in x for x in r.not_enforced), (
        "the per-cell composition table must be printed, not merely asserted"
    )
