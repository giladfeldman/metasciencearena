"""Tests for the tool-feedback report engine."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from framework.report import _improvement_priorities, generate_report, render_markdown


def _make_arena_dir(tmp_path: Path, *, with_categories: bool = True) -> Path:
    arena = tmp_path / "synth-arena-v1"
    (arena / "schemas").mkdir(parents=True)
    (arena / "task_sets" / "v1").mkdir(parents=True)
    (arena / "runs" / "v1").mkdir(parents=True)
    manifest = {
        "arena_id": "synth-arena-v1",
        "taxonomy_leaf_id": "synth-leaf",
        "version": "0.0.1",
        "owner": "tests",
        "license": "MIT",
        "input_artifact": "plain_text_section",
        "output_to_judge": "classification_label",
        "difficulty_axes": [{"id": "tier", "min": 1, "max": 3, "description": "Test tier"}],
        "task_set_versions": [{"version": "v1", "n_tasks": 4, "released_on": "2026-05-06"}],
        "display_columns": [
            {"id": "composite", "source": "score.primary", "label": "Composite",
             "description": "Test composite", "direction": "higher", "format": "f3", "primary": True},
        ],
        "tier_pivot": {
            "axis": "tier",
            "composite_formula": "test",
            "values": [
                {"value": 1, "label": "T1", "sub": "one"},
                {"value": 2, "label": "T2", "sub": "two"},
                {"value": 3, "label": "T3", "sub": "three"},
            ],
        },
    }
    if with_categories:
        manifest["error_categories"] = [
            {"id": "off_by_one", "description": "Off by one error.", "severity": "minor"},
            {"id": "wrong_label", "description": "Wrong classification label.", "severity": "major"},
        ]
    (arena / "arena.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    # Output schema (required by load_arena indirectly when running, but optional here).
    (arena / "schemas" / "input.schema.json").write_text(json.dumps({"type": "object"}), encoding="utf-8")
    (arena / "schemas" / "output.schema.json").write_text(json.dumps({"type": "object"}), encoding="utf-8")
    # Stub generator + scorer files (load_arena requires neither to exist for report).
    (arena / "generator.py").write_text("def generate(*a, **k): return iter([])\n", encoding="utf-8")
    (arena / "scorer.py").write_text("def score(*a, **k): return {'primary': 1.0, 'breakdown': {}}\n", encoding="utf-8")
    return arena


def _record(**overrides) -> dict:
    base = {
        "run_id": "r-x", "arena_id": "synth-arena-v1", "task_set_version": "v1",
        "task_id": "t1", "player_id": "p", "player_version": "0.1.0", "player_type": "tool",
        "input_hash": "0" * 64, "output": {"label": "ok"},
        "score": {"primary": 0.7, "breakdown": {}},
        "task_visibility": "public",
        "timestamp_utc": "2026-05-06T12:00:00Z",
    }
    base.update(overrides)
    return base


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, sort_keys=True) + "\n")


# ----------------------------------------------------------------------------
# Core happy paths
# ----------------------------------------------------------------------------


def test_generate_report_writes_bundle_with_expected_files(tmp_path):
    arena = _make_arena_dir(tmp_path)
    records = [
        _record(task_id="t1", score={"primary": 0.8, "breakdown": {},
                                     "findings": [{"category": "off_by_one"}]}),
        _record(task_id="t2", score={"primary": 0.6, "breakdown": {},
                                     "findings": [{"category": "wrong_label", "evidence": "bad"}]}),
        _record(task_id="t3", task_visibility="held_out",
                score={"primary": 0.4, "breakdown": {},
                       "findings": [{"category": "off_by_one"}]}),
    ]
    _write_jsonl(arena / "runs" / "v1" / "main.jsonl", records)

    bundle = generate_report(
        arena_dir=arena, task_set_version="v1", player_id="p", player_version="0.1.0",
    )
    assert (bundle / "tool_report.json").is_file()
    assert (bundle / "tool_report.md").is_file()
    assert (bundle / "README.md").is_file()
    assert (bundle / "findings" / "public" / "t1.json").is_file()
    assert (bundle / "findings" / "public" / "t2.json").is_file()
    assert (bundle / "findings" / "held_out" / "t3.json").is_file()


def test_summary_block_has_rank_and_means(tmp_path):
    arena = _make_arena_dir(tmp_path)
    records = [
        _record(player_id="me", task_id="t1", score={"primary": 0.9, "breakdown": {}}),
        _record(player_id="me", task_id="t2", score={"primary": 0.7, "breakdown": {}}),
        _record(player_id="them", task_id="t1", score={"primary": 0.5, "breakdown": {}}),
        _record(player_id="them", task_id="t2", score={"primary": 0.5, "breakdown": {}}),
    ]
    _write_jsonl(arena / "runs" / "v1" / "main.jsonl", records)
    bundle = generate_report(
        arena_dir=arena, task_set_version="v1", player_id="me", player_version="0.1.0",
    )
    report = json.loads((bundle / "tool_report.json").read_text(encoding="utf-8"))
    assert report["summary"]["n_competitors"] == 2
    assert report["summary"]["rank"] == 1
    assert 0.79 < report["summary"]["primary_mean"] < 0.81


def test_error_histogram_sorted_by_findings(tmp_path):
    arena = _make_arena_dir(tmp_path)
    records = [
        _record(task_id="t1", score={"primary": 0.6, "breakdown": {},
                                     "findings": [{"category": "off_by_one", "count": 5}]}),
        _record(task_id="t2", score={"primary": 0.6, "breakdown": {},
                                     "findings": [{"category": "wrong_label"},
                                                  {"category": "off_by_one", "count": 2}]}),
    ]
    _write_jsonl(arena / "runs" / "v1" / "main.jsonl", records)
    bundle = generate_report(
        arena_dir=arena, task_set_version="v1", player_id="p", player_version="0.1.0",
    )
    report = json.loads((bundle / "tool_report.json").read_text(encoding="utf-8"))
    hist = report["error_histogram"]
    assert hist[0]["category"] == "off_by_one"     # 5 + 2 = 7 findings
    assert hist[0]["n_findings"] == 7
    assert hist[0]["n_tasks_affected"] == 2
    assert hist[0]["severity"] == "minor"
    assert hist[1]["category"] == "wrong_label"
    assert hist[1]["n_findings"] == 1
    assert hist[1]["severity"] == "major"


# ----------------------------------------------------------------------------
# Held-out redaction
# ----------------------------------------------------------------------------


def test_held_out_drilldown_contains_no_content(tmp_path):
    arena = _make_arena_dir(tmp_path)
    # Held-out finding with content fields — simulating a malformed record that
    # somehow carries gold content through. Report engine must NOT pass it
    # through to the held_out drilldown file. (Note: in production the runner
    # strips before write, but the report engine itself should be defensive.)
    records = [
        _record(task_id="ho-1", task_visibility="held_out",
                score={"primary": 0.4, "breakdown": {"levenshtein_similarity": 0.4},
                       "findings": [{"category": "off_by_one", "count": 3}]}),
    ]
    _write_jsonl(arena / "runs" / "v1" / "main.jsonl", records)
    bundle = generate_report(
        arena_dir=arena, task_set_version="v1", player_id="p", player_version="0.1.0",
    )
    ho = json.loads((bundle / "findings" / "held_out" / "ho-1.json").read_text(encoding="utf-8"))
    assert "gold" not in ho
    assert "your_output" not in ho
    assert "input_hash" not in ho
    assert "best_player_score" not in ho
    # breakdown is preserved (numeric only by contract).
    assert ho["your_score"]["breakdown"]["levenshtein_similarity"] == 0.4
    # No public-tasks or reproduce envelopes were written for the held-out task.
    assert not (bundle / "findings" / "public" / "ho-1.json").exists()
    assert not (bundle / "reproduce" / "ho-1.json").exists()


def test_held_out_aggregate_separated_from_public(tmp_path):
    arena = _make_arena_dir(tmp_path)
    records = [
        _record(task_id="t1", task_visibility="public",
                score={"primary": 0.9, "breakdown": {}}),
        _record(task_id="ho-1", task_visibility="held_out",
                score={"primary": 0.3, "breakdown": {}}),
        _record(task_id="ho-2", task_visibility="held_out",
                score={"primary": 0.4, "breakdown": {}}),
    ]
    _write_jsonl(arena / "runs" / "v1" / "main.jsonl", records)
    bundle = generate_report(
        arena_dir=arena, task_set_version="v1", player_id="p", player_version="0.1.0",
    )
    report = json.loads((bundle / "tool_report.json").read_text(encoding="utf-8"))
    assert report["held_out_aggregate"]["n_scored"] == 2
    assert 0.34 < report["held_out_aggregate"]["primary_mean"] < 0.36
    # Public drilldown only mentions t1, never the held-out tasks.
    pub_ids = {entry["task_id"] for entry in report["public_drilldown_index"]}
    assert pub_ids == {"t1"}


# ----------------------------------------------------------------------------
# Empty / shallow cases
# ----------------------------------------------------------------------------


def test_arena_without_categories_produces_shallow_report(tmp_path):
    arena = _make_arena_dir(tmp_path, with_categories=False)
    records = [
        _record(task_id="t1", score={"primary": 0.8, "breakdown": {}}),
    ]
    _write_jsonl(arena / "runs" / "v1" / "main.jsonl", records)
    bundle = generate_report(
        arena_dir=arena, task_set_version="v1", player_id="p", player_version="0.1.0",
    )
    report = json.loads((bundle / "tool_report.json").read_text(encoding="utf-8"))
    assert report["error_histogram"] == []
    # Summary still computed.
    assert report["summary"]["n_scored"] == 1


def test_no_records_for_player_raises(tmp_path):
    arena = _make_arena_dir(tmp_path)
    _write_jsonl(arena / "runs" / "v1" / "main.jsonl", [_record(player_id="someone-else")])
    with pytest.raises(ValueError):
        generate_report(
            arena_dir=arena, task_set_version="v1", player_id="ghost", player_version="0.0.0",
        )


# ----------------------------------------------------------------------------
# Version diff
# ----------------------------------------------------------------------------


def test_version_diff_when_previous_version_exists(tmp_path):
    arena = _make_arena_dir(tmp_path)
    records = [
        # Previous version: lots of off_by_one
        _record(player_version="0.0.9", task_id="t1",
                score={"primary": 0.5, "breakdown": {},
                       "findings": [{"category": "off_by_one", "count": 5}]}),
        _record(player_version="0.0.9", task_id="t2",
                score={"primary": 0.5, "breakdown": {},
                       "findings": [{"category": "off_by_one", "count": 5}]}),
        # Current version: fewer off_by_one, but introduced wrong_label
        _record(player_version="0.1.0", task_id="t1",
                score={"primary": 0.7, "breakdown": {},
                       "findings": [{"category": "off_by_one", "count": 1}]}),
        _record(player_version="0.1.0", task_id="t2",
                score={"primary": 0.8, "breakdown": {},
                       "findings": [{"category": "wrong_label"}]}),
    ]
    _write_jsonl(arena / "runs" / "v1" / "main.jsonl", records)
    bundle = generate_report(
        arena_dir=arena, task_set_version="v1", player_id="p", player_version="0.1.0",
    )
    report = json.loads((bundle / "tool_report.json").read_text(encoding="utf-8"))
    vd = report["version_diff"]
    assert vd is not None
    assert vd["previous_version"] == "0.0.9"
    assert vd["primary_delta"] > 0
    assert "off_by_one" in vd["categories_improved"]
    assert "wrong_label" in vd["categories_regressed"]


def test_version_diff_picks_semver_predecessor_not_lexicographic_max(tmp_path):
    """0.10.0 must compare > 0.9.0 and > 0.2.0 — lexicographic sort would
    pick 0.9.0 as previous, semver-aware must pick 0.9.0 as the highest
    version below the focal 0.10.0."""
    arena = _make_arena_dir(tmp_path)
    records = [
        _record(player_version="0.2.0", task_id="t1",
                score={"primary": 0.4, "breakdown": {}, "findings": [{"category": "off_by_one"}]}),
        _record(player_version="0.9.0", task_id="t1",
                score={"primary": 0.6, "breakdown": {}, "findings": [{"category": "off_by_one"}]}),
        _record(player_version="0.10.0", task_id="t1",
                score={"primary": 0.8, "breakdown": {}}),
    ]
    _write_jsonl(arena / "runs" / "v1" / "main.jsonl", records)
    bundle = generate_report(
        arena_dir=arena, task_set_version="v1", player_id="p", player_version="0.10.0",
    )
    report = json.loads((bundle / "tool_report.json").read_text(encoding="utf-8"))
    assert report["version_diff"]["previous_version"] == "0.9.0"


def test_version_diff_null_for_first_submission(tmp_path):
    arena = _make_arena_dir(tmp_path)
    records = [_record(task_id="t1", score={"primary": 0.7, "breakdown": {}})]
    _write_jsonl(arena / "runs" / "v1" / "main.jsonl", records)
    bundle = generate_report(
        arena_dir=arena, task_set_version="v1", player_id="p", player_version="0.1.0",
    )
    report = json.loads((bundle / "tool_report.json").read_text(encoding="utf-8"))
    assert report["version_diff"] is None


# ----------------------------------------------------------------------------
# Markdown rendering smoke test
# ----------------------------------------------------------------------------


def test_render_markdown_includes_key_sections():
    report = {
        "report_id": "uuid",
        "generated_at_utc": "2026-05-06T12:00:00Z",
        "arena": {"id": "x", "task_set_version": "v1"},
        "player": {"id": "p", "version": "0.1.0"},
        "summary": {"primary_mean": 0.7, "primary_ci_half": 0.05, "n_scored": 2,
                    "n_excluded": 0, "n_errored": 0, "rank": 1, "n_competitors": 1,
                    "cost_usd_mean": None, "latency_ms_mean": 100},
        "version_diff": None,
        "error_histogram": [],
        "difficulty_breakdown": {"by_axis": {}},
        "vs_competitors": [],
        "public_drilldown_index": [],
        "held_out_aggregate": {"n_scored": 0, "primary_mean": None, "category_histogram": []},
    }
    md = render_markdown(report)
    assert "Tool Report" in md
    assert "## Summary" in md
    assert "Held-out aggregate" in md
    assert "redacted by construction" in md


# ----------------------------------------------------------------------------
# _improvement_priorities
# ----------------------------------------------------------------------------


def _fixture_inputs():
    return dict(
        public_hist=[
            {"category": "footnote_missed", "n_findings": 80, "n_tasks_affected": 24,
             "share_of_findings": 0.50, "severity": "major"},
            {"category": "greek_letters_dropped", "n_findings": 60, "n_tasks_affected": 10,
             "share_of_findings": 0.40, "severity": "minor"},
        ],
        held_out_hist=[
            {"category": "footnote_missed", "n_findings": 10, "n_tasks_affected": 5,
             "share_of_findings": 0.20, "severity": "major"},
            {"category": "greek_letters_dropped", "n_findings": 40, "n_tasks_affected": 9,
             "share_of_findings": 0.70, "severity": "minor"},
        ],
        public_mean=0.62,
        held_out_mean=0.34,
        diff_breakdown={"by_axis": {
            "source_kind": [
                {"value": 1, "n_scored": 48, "primary_mean": 0.90},
                {"value": 2, "n_scored": 47, "primary_mean": 0.34},
            ],
            "tier": [{"value": 1, "n_scored": 95, "primary_mean": 0.62}],
        }},
        competitors=[
            {"competitor_id": "grobid-text-only", "competitor_version": "grobid-0.8.x",
             "their_primary_mean": 0.81, "tasks_where_they_won": 46, "tasks_where_you_won": 8,
             "your_dominant_failure_mode_on_their_wins": "footnote_missed",
             "their_dominant_failure_mode_on_your_wins": "footnote_missed"},
        ],
        cat_examples={"footnote_missed": ["synth-01-c0", "synth-02-c1"],
                      "greek_letters_dropped": ["synth-03-c0"]},
        competitor_examples={"grobid-text-only": ["synth-02-c1", "synth-05-c2"]},
    )


def test_priorities_ranks_and_frames():
    items = _improvement_priorities(**_fixture_inputs())
    kinds = [i["kind"] for i in items]
    assert kinds[0] == "failure_mode"
    assert "difficulty_axis" in kinds
    assert "beaten_by" in kinds
    assert "hidden_gap" in kinds
    assert [i["rank"] for i in items] == list(range(1, len(items) + 1))
    top = items[0]
    assert top["evidence"]["category"] == "footnote_missed"
    assert top["public_examples"] == ["synth-01-c0", "synth-02-c1"]
    assert "footnote_missed" in top["headline"]


def test_priorities_hidden_gap_has_no_examples():
    items = _improvement_priorities(**_fixture_inputs())
    hidden = [i for i in items if i["kind"] == "hidden_gap"]
    assert hidden, "expected a hidden_gap item (greek share 0.40->0.70)"
    for h in hidden:
        assert "public_examples" not in h
        assert "task" not in h["headline"].lower()
        assert set(h["evidence"]).issubset({"category", "public_share", "held_out_share",
                                            "public_mean", "held_out_mean", "signal"})


def test_priorities_empty_when_no_signal():
    assert _improvement_priorities(
        public_hist=[], held_out_hist=[], public_mean=0.9, held_out_mean=0.9,
        diff_breakdown={"by_axis": {}}, competitors=[], cat_examples={},
        competitor_examples={}) == []


def test_generate_report_includes_priorities_key():
    from framework import report as R
    sample = {
        "summary": {"primary_mean": 0.6, "primary_ci_half": 0.05, "n_scored": 10,
                    "n_excluded": 0, "n_errored": 0, "rank": 2, "n_competitors": 3,
                    "cost_usd_mean": None, "latency_ms_mean": 100.0},
        "improvement_priorities": [
            {"rank": 1, "kind": "failure_mode", "headline": "Reduce `x`: 50% of findings.",
             "evidence": {"category": "x"}, "public_examples": ["t1"]}],
        "error_histogram": [], "difficulty_breakdown": {"by_axis": {}},
        "vs_competitors": [], "public_drilldown_index": [],
        "held_out_aggregate": {"n_scored": 0, "primary_mean": None, "category_histogram": []},
        "version_diff": None,
        "arena": {"id": "a", "task_set_version": "v1"},
        "player": {"id": "p", "version": "1"},
    }
    md = R.render_markdown(sample)
    assert "Improvement priorities" in md
    assert "Reduce `x`" in md


def test_hidden_gap_items_never_leak_examples():
    from framework.report import _improvement_priorities
    items = _improvement_priorities(
        public_hist=[{"category": "x", "n_findings": 1, "n_tasks_affected": 1,
                      "share_of_findings": 0.1, "severity": "minor"}],
        held_out_hist=[{"category": "x", "n_findings": 9, "n_tasks_affected": 5,
                        "share_of_findings": 0.9, "severity": "minor"}],
        public_mean=0.9, held_out_mean=0.3,
        diff_breakdown={"by_axis": {}}, competitors=[],
        cat_examples={"x": ["t1"]}, competitor_examples={})
    for h in [i for i in items if i["kind"] == "hidden_gap"]:
        assert "public_examples" not in h
        assert "t1" not in str(h)


# ----------------------------------------------------------------------------
# Regression: cost must be DERIVED from usage, not read from a field nothing writes.
#
# Found 2026-08-14. `framework/pricing.py` states the design plainly — "A run
# record stores token counts... So cost is derived here, at report time" — and
# `cost_usd` is therefore deliberately never written to a record. But `_summary`
# read `r["cost_usd"]` off the record, and `report.mjs` (the Node build that
# actually produces the PUBLISHED reports) did the same. Both consumers read a
# field both producers intentionally leave absent, so `cost_usd_mean` was `null`
# in all 127 published reports while the pricing module sat fully unit-tested
# and entirely uncalled.
#
# This is the seam defect CLAUDE.md describes: every unit correct, the value
# never arriving. The test asserts arrival, not unit behaviour.
# ----------------------------------------------------------------------------


def test_cost_usd_mean_is_derived_from_usage(tmp_path):
    arena = _make_arena_dir(tmp_path)
    # A priced model (in pricing.PRICES) with recorded token usage, and NO
    # cost_usd key — exactly what the runner writes.
    usage = {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}
    records = [
        _record(task_id="t1", player_type="ai-model",
                resolved_tool_version="openai/gpt-oss-120b", usage=dict(usage),
                score={"primary": 0.8, "breakdown": {}}),
        _record(task_id="t2", player_type="ai-model",
                resolved_tool_version="openai/gpt-oss-120b", usage=dict(usage),
                score={"primary": 0.6, "breakdown": {}}),
    ]
    _write_jsonl(arena / "runs" / "v1" / "main.jsonl", records)

    bundle = generate_report(
        arena_dir=arena, task_set_version="v1", player_id="p", player_version="0.1.0",
    )
    summary = json.loads((bundle / "tool_report.json").read_text(encoding="utf-8"))["summary"]

    # gpt-oss-120b = $0.15/1M in + $0.75/1M out -> $0.90 per task at 1M+1M.
    assert summary["cost_usd_mean"] is not None, (
        "cost_usd_mean is None despite priced records carrying usage — the "
        "pricing module is not being called"
    )
    assert summary["cost_usd_mean"] == pytest.approx(0.90)


def test_cost_usd_mean_stays_none_for_subscription_players(tmp_path):
    """A Claude-via-Claude-Max run consumes tokens but has no per-token price.
    None is the honest answer; 0.0 would claim the run was free."""
    arena = _make_arena_dir(tmp_path)
    records = [
        _record(task_id="t1", player_id="claude-opus-4-8", player_type="ai-model",
                player_version="claude-opus-4-8",
                usage={"prompt_tokens": 1000, "completion_tokens": 500},
                score={"primary": 0.8, "breakdown": {}}),
    ]
    _write_jsonl(arena / "runs" / "v1" / "main.jsonl", records)
    bundle = generate_report(
        arena_dir=arena, task_set_version="v1",
        player_id="claude-opus-4-8", player_version="claude-opus-4-8",
    )
    summary = json.loads((bundle / "tool_report.json").read_text(encoding="utf-8"))["summary"]
    assert summary["cost_usd_mean"] is None
    assert summary["tokens_total"] == 1500, "tokens must still be reported for unpriced players"


def test_containment_describes_the_focal_player_not_the_arena(tmp_path):
    """Regression, 2026-08-19: `containment` was summarised over ALL arena records.

    `_summary` was passed `all_records` rather than `focal_records`, so a single
    agentic-CLI competitor labelled every other player in the arena
    ``uncontrolled`` — including R tools and HTTP players that have no ambient
    environment to inherit. Caught by counting the published artifacts: the arena
    manifests reported 58 of 130 result sets uncontrolled while the reports built
    from the same records reported 123. The manifests were right.
    """
    arena = _make_arena_dir(tmp_path)
    records = [
        _record(player_id="rtool", task_id="t1", score={"primary": 0.9, "breakdown": {}},
                provenance={"adapter_class": "RScriptAdapter"}),
        _record(player_id="rtool", task_id="t2", score={"primary": 0.8, "breakdown": {}},
                provenance={"adapter_class": "RScriptAdapter"}),
        _record(player_id="cliplayer", task_id="t1", score={"primary": 0.5, "breakdown": {}},
                provenance={"adapter_class": "SubprocessCliAdapter"}),
        _record(player_id="cliplayer", task_id="t2", score={"primary": 0.4, "breakdown": {}},
                provenance={"adapter_class": "SubprocessCliAdapter"}),
    ]
    _write_jsonl(arena / "runs" / "v1" / "main.jsonl", records)

    bundle = generate_report(arena_dir=arena, task_set_version="v1",
                             player_id="rtool", player_version="0.1.0")
    report = json.loads((bundle / "tool_report.json").read_text(encoding="utf-8"))
    assert report["summary"]["containment"]["worst"] == "not-applicable"
    assert "uncontrolled" not in report["summary"]["containment"]["states"]

    bundle = generate_report(arena_dir=arena, task_set_version="v1",
                             player_id="cliplayer", player_version="0.1.0")
    report = json.loads((bundle / "tool_report.json").read_text(encoding="utf-8"))
    assert report["summary"]["containment"]["worst"] == "uncontrolled"
