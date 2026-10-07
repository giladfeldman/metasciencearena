"""A task set may ask a different question from its predecessor.

publication-bias-v1 v2 asks for every method's statistics where v1 asked for a
verdict, so the OUTPUT SCHEMA an answer is validated against belongs to the task
set, not to the arena. Before 2026-09-30 it was fixed per arena: a v2 run would have
rejected every correct v2 answer as a schema violation. (The prompt needs no such
mechanism: the runner only ever plays the CURRENT task set, the registry names that
set's prompt, and a superseded prompt is archived under players/prompts/_archive/
so every published record's hash still resolves.)
"""
from pathlib import Path

import pytest

from framework.discovery import load_arena, output_schema_path_for

REPO_ROOT = Path(__file__).resolve().parents[2]
PUBBIAS = REPO_ROOT / "arenas" / "publication-bias-v1"


def test_a_version_with_its_own_schema_gets_it_and_others_fall_back():
    arena = load_arena(PUBBIAS)
    assert output_schema_path_for(arena, "v2") == PUBBIAS / "schemas" / "v2" / "output.schema.json"
    assert output_schema_path_for(arena, "v1") == PUBBIAS / "schemas" / "output.schema.json"
    assert output_schema_path_for(arena, None) == PUBBIAS / "schemas" / "output.schema.json"


def test_an_arena_without_per_version_schemas_is_unchanged():
    arena = load_arena(REPO_ROOT / "arenas" / "grim-consistency-v1")
    assert output_schema_path_for(arena, "v1") == Path(arena["output_schema_path"])


def test_the_runner_validates_against_the_played_versions_schema(monkeypatch, tmp_path):
    """Drive run_tournament far enough to see what it built, without playing."""
    import framework.runner as runner

    seen = {}

    class Stop(Exception):
        pass

    def fake_build_adapter(entry):
        seen["prompt"] = entry.get("prompt_template_path")
        raise Stop

    real_validator = runner.Draft202012Validator

    def spy_validator(schema, *a, **k):
        seen.setdefault("schema_title", schema.get("title"))
        return real_validator(schema, *a, **k)

    monkeypatch.setattr(runner, "build_adapter", fake_build_adapter)
    monkeypatch.setattr(runner, "Draft202012Validator", spy_validator)
    monkeypatch.setattr(runner, "assert_heldout_egress_allowed", lambda *a, **k: None)
    with pytest.raises(Stop):
        runner.run_tournament(arena_dir=PUBBIAS, task_set_version="v2",
                              registry_path=REPO_ROOT / "players" / "registry.yaml",
                              player_ids=["claude-opus-5-pubbias"],
                              output_path=tmp_path / "out.jsonl", public_only=True)
    assert seen["prompt"] == "players/prompts/publication_bias_v2.txt"
    assert "task set v2" in seen["schema_title"]
