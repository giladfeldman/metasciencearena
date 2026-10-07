"""Tests for the tournament runner."""
import json
from framework.runner import _redact_findings_for_held_out, run_tournament
from framework.storage import read_records


def test_runner_writes_records_for_pass_and_fail(tmp_path, fake_arena_dir, fake_registry_path):
    out = tmp_path / "runs.jsonl"
    n = run_tournament(
        arena_dir=fake_arena_dir,
        task_set_version="v1",
        registry_path=fake_registry_path,
        player_ids=["stub-pass", "stub-fail"],
        output_path=out,
        trials=1,
    )
    records = read_records(out)
    by_player = {r["player_id"] for r in records}
    assert by_player == {"stub-pass", "stub-fail"}
    # 2 tasks × 2 players × 1 trial = 4
    assert len(records) == 4 == n
    # stub-fail should have score 0 with error breakdown
    fails = [r for r in records if r["player_id"] == "stub-fail"]
    assert all(r["score"]["primary"] == 0.0 for r in fails)
    assert all("error" in r["score"]["breakdown"] for r in fails)


def test_runner_forces_one_trial_for_deterministic(tmp_path, fake_arena_dir, fake_registry_path):
    out = tmp_path / "runs.jsonl"
    run_tournament(
        arena_dir=fake_arena_dir,
        task_set_version="v1",
        registry_path=fake_registry_path,
        player_ids=["stub-pass"],
        output_path=out,
        trials=5,  # deterministic player should still produce 1 trial per task
    )
    records = read_records(out)
    assert len(records) == 2  # 2 tasks × 1 trial (forced)


def test_runner_writes_task_visibility_default_held_out(tmp_path, fake_arena_dir, fake_registry_path):
    """Generators that don't set envelope.visibility default to held_out (fail-safe)."""
    out = tmp_path / "runs.jsonl"
    run_tournament(
        arena_dir=fake_arena_dir,
        task_set_version="v1",
        registry_path=fake_registry_path,
        player_ids=["stub-pass"],
        output_path=out,
        trials=1,
    )
    records = read_records(out)
    assert all(r["task_visibility"] == "held_out" for r in records)


def test_redact_findings_for_held_out_strips_content_fields():
    findings = [
        {"category": "off_by_one", "anchor": {"page": 1}, "evidence": "x",
         "correct_value": "y", "count": 3, "examples": ["a", "b"]},
        {"category": "wrong_label", "evidence": "wrong"},
    ]
    redacted = _redact_findings_for_held_out(findings)
    assert redacted == [
        {"category": "off_by_one", "count": 3},
        {"category": "wrong_label"},
    ]


def test_redact_findings_drops_findings_without_category():
    redacted = _redact_findings_for_held_out([{"evidence": "no category"}])
    assert redacted == []


def test_redact_findings_handles_non_list():
    assert _redact_findings_for_held_out(None) is None
    assert _redact_findings_for_held_out("not a list") == "not a list"


def test_revealed_split_records_split_and_keeps_findings(tmp_path, fake_split_arena_dir, fake_registry_path):
    out = tmp_path / "rev.jsonl"
    run_tournament(
        arena_dir=fake_split_arena_dir,
        task_set_version="v1",
        registry_path=fake_registry_path,
        player_ids=["stub-pass"],
        output_path=out,
        trials=1,
        split="revealed",
    )
    [rec] = read_records(out)
    assert rec["split"] == "revealed"
    assert rec["task_visibility"] == "public"
    # Revealed findings keep their content-bearing fields (nothing redacted).
    finding = rec["score"]["findings"][0]
    assert finding["category"] == "wrong_label"
    assert finding["evidence"] == "ok"
    assert finding["correct_value"] == "gold"


def test_private_split_records_split_and_redacts_findings(tmp_path, fake_split_arena_dir, fake_registry_path):
    out = tmp_path / "priv.jsonl"
    run_tournament(
        arena_dir=fake_split_arena_dir,
        task_set_version="v1",
        registry_path=fake_registry_path,
        player_ids=["stub-pass"],
        output_path=out,
        trials=1,
        split="private",
    )
    [rec] = read_records(out)
    assert rec["split"] == "private"
    assert rec["task_visibility"] == "held_out"
    # Private findings are redacted to category-only (content stripped before write).
    assert rec["score"]["findings"] == [{"category": "wrong_label"}]


def test_public_only_and_held_out_only_filters(tmp_path, fake_arena_dir, fake_registry_path):
    """fake_arena_dir's tasks default to held_out. --public-only drops them all;
    --held-out-only keeps them all (mirror filters, Finding 2 split hygiene)."""
    pub = tmp_path / "pub.jsonl"
    n_pub = run_tournament(
        arena_dir=fake_arena_dir, task_set_version="v1", registry_path=fake_registry_path,
        player_ids=["stub-pass"], output_path=pub, trials=1, public_only=True,
    )
    assert n_pub == 0  # every task is held_out

    held = tmp_path / "held.jsonl"
    n_held = run_tournament(
        arena_dir=fake_arena_dir, task_set_version="v1", registry_path=fake_registry_path,
        player_ids=["stub-pass"], output_path=held, trials=1, held_out_only=True,
    )
    records = read_records(held)
    assert n_held == len(records) > 0
    assert all(r["task_visibility"] == "held_out" for r in records)


def test_legacy_generator_without_split_omits_split_field(tmp_path, fake_arena_dir, fake_registry_path):
    """A generator whose signature lacks `split` still runs; records omit `split`."""
    out = tmp_path / "legacy.jsonl"
    run_tournament(
        arena_dir=fake_arena_dir,
        task_set_version="v1",
        registry_path=fake_registry_path,
        player_ids=["stub-pass"],
        output_path=out,
        trials=1,
        split="private",  # ignored — generator doesn't accept it
    )
    records = read_records(out)
    assert records
    assert all("split" not in r for r in records)
