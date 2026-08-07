"""Tests for the revealed/private parity checker."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from framework import parity

REPO_ROOT = Path(__file__).resolve().parents[2]
STATS_ARENA = REPO_ROOT / "arenas" / "stats-extraction-v1"


def test_mistake_labels_marks_clean_vs_injected():
    gold = {"items": [
        {"kind": "nhst_stat", "deception_kind": None},
        {"kind": "nhst_stat", "deception_kind": "impossible_p"},
    ]}
    assert parity._mistake_labels(gold) == ["clean", "impossible_p"]


def test_mistake_labels_handles_deviation_kind_key():
    gold = {"items": [{"deviation_kind": "outcome_switch"}, {"deviation_kind": None}]}
    assert parity._mistake_labels(gold) == ["outcome_switch", "clean"]


def test_mistake_labels_prefers_top_level_mistake_kinds_list():
    # Field-map-style gold (no per-item *_kind) declares mistakes at the top level.
    gold = {"has_power_analysis": True, "kind": "posthoc",
            "mistake_kinds": ["posthoc_as_apriori", "missing_field"]}
    assert parity._mistake_labels(gold) == ["posthoc_as_apriori", "missing_field"]


def test_mistake_labels_empty_top_level_is_clean():
    assert parity._mistake_labels({"fields": {}, "mistake_kinds": []}) == ["clean"]


def test_within_tolerance():
    assert parity._within_tolerance(10, 10, 0.0)
    assert not parity._within_tolerance(10, 9, 0.0)
    assert parity._within_tolerance(10, 9, 0.1)
    assert not parity._within_tolerance(10, 8, 0.1)


def test_resolve_seed_revealed_from_manifest(tmp_path):
    manifest = {"benchmark_splits": {"revealed": {"seed": 0},
                                     "private": {"seed_source": "secret_file"}}}
    seed, note = parity.resolve_seed(manifest, tmp_path, "v1", "revealed")
    assert seed == 0 and note is None


def test_resolve_seed_private_secret_file(tmp_path):
    manifest = {"benchmark_splits": {"revealed": {"seed": 0},
                                     "private": {"seed_source": "secret_file"}}}
    seed_dir = tmp_path / "task_sets" / "v1"
    seed_dir.mkdir(parents=True)
    (seed_dir / ".private_seed").write_text("424242", encoding="utf-8")
    seed, note = parity.resolve_seed(manifest, tmp_path, "v1", "private")
    assert seed == 424242 and note is None


def test_resolve_seed_private_env(tmp_path, monkeypatch):
    manifest = {"benchmark_splits": {"revealed": {"seed": 0},
                                     "private": {"seed_source": "env"}}}
    monkeypatch.setenv("SCIENCEARENA_PRIVATE_SEED", "777")
    seed, note = parity.resolve_seed(manifest, tmp_path, "v1", "private")
    assert seed == 777 and note is None


def test_resolve_seed_private_dev_fallback_warns(tmp_path):
    manifest = {"benchmark_splits": {"revealed": {"seed": 5},
                                     "private": {"seed_source": "secret_file"}}}
    seed, note = parity.resolve_seed(manifest, tmp_path, "v1", "private")
    assert seed == 5 + parity._DEV_PRIVATE_OFFSET
    assert note is not None and "dev-fallback" in note


def test_stats_extraction_arena_passes_parity():
    report = parity.check_parity(STATS_ARENA, "v1")
    assert report.ok, report.summary()
    # Both splits should cover the same difficulty cells and mistake categories.
    assert set(report.revealed_cells) == set(report.private_cells)
    assert set(report.revealed_categories) == set(report.private_categories)
    # Whether the private seed comes from a committed .private_seed (real secret,
    # gitignored) or the dev fallback is environment-dependent — do NOT assert on
    # which one. The fallback mechanism itself is covered by
    # test_resolve_seed_private_dev_fallback_warns. The invariant that always holds
    # is that the private suite is NOT identical to the public one.
    assert not any("SAME seed" in p for p in report.problems)


def test_single_suite_arena_is_skipped(tmp_path):
    arena_dir = tmp_path / "no-splits-arena"
    arena_dir.mkdir()
    (arena_dir / "arena.yaml").write_text("arena_id: no-splits-arena\n", encoding="utf-8")
    report = parity.check_parity(arena_dir, "v1")
    assert report.ok
    assert any("single-suite" in n for n in report.notes)


def test_parity_detects_category_mismatch(tmp_path, monkeypatch):
    """A generator whose private split drops a mistake category must FAIL parity."""
    arena_dir = tmp_path / "skewed-arena"
    arena_dir.mkdir()
    (arena_dir / "arena.yaml").write_text(
        "arena_id: skewed-arena\n"
        "benchmark_splits:\n"
        "  revealed: {seed: 0}\n"
        "  private: {seed_source: env}\n"
        "  parity:\n"
        "    match_axes: [tier]\n"
        "    match_categories: true\n"
        "    count_tolerance: 0.0\n",
        encoding="utf-8",
    )
    # A generator that injects an extra mistake category only in the revealed split.
    (arena_dir / "generator.py").write_text(
        "_GROUND_TRUTH_CACHE = {}\n"
        "def generate(task_set_version, seed, split='revealed'):\n"
        "    _GROUND_TRUTH_CACHE.clear()\n"
        "    for tier in (1, 2):\n"
        "        tid = f'{split}-t{tier}'\n"
        "        kind = 'extra_bug' if (split == 'revealed' and tier == 2) else 'common_bug'\n"
        "        _GROUND_TRUTH_CACHE[tid] = {'items': [{'deviation_kind': kind}]}\n"
        "        yield {'task_id': tid, 'difficulty': {'tier': tier},\n"
        "               'split': split, 'visibility': 'public' if split=='revealed' else 'held_out'}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SCIENCEARENA_PRIVATE_SEED", "1")
    report = parity.check_parity(arena_dir, "v1")
    assert not report.ok
    assert any("only in revealed" in p for p in report.problems)


def test_independent_holdout_skips_cell_parity(tmp_path, monkeypatch):
    """A real, separate holdout corpus cannot mirror the revealed grid.

    Added 2026-08-04 with code-translation-r-v1's real `_held_out/` corpus. Real
    third-party scripts arrive with whatever constructs their authors used, so
    cell/category parity is unenforceable — and hand-picking real material to
    fit the template would stop it being a holdout. The contract now models this
    explicitly (`parity.independent_holdout`) instead of leaving an arena to
    fake a tolerance wide enough to pass.
    """
    from framework.parity import check_parity

    arena = tmp_path / "fake-independent-v1"
    (arena / "task_sets" / "v1").mkdir(parents=True)
    (arena / "arena.yaml").write_text(
        "arena_id: fake-independent-v1\n"
        "benchmark_splits:\n"
        "  revealed: {seed: 0, published: true}\n"
        "  private: {seed_source: secret_file}\n"
        "  parity:\n"
        "    independent_holdout: true\n"
        "    match_axes: [tier]\n",
        encoding="utf-8")

    rep = check_parity(arena, "v1")
    assert rep.ok, rep.problems
    assert any("independent-holdout" in n for n in rep.notes), rep.notes
    # It must SAY it skipped, not silently pass.
    assert not rep.problems
