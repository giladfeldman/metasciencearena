"""Task set v2: null summaries are valid, ids are versioned, private ids are seed-free (d-603203)."""
import importlib.util
import json
from pathlib import Path

from jsonschema import Draft202012Validator

ARENA = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("pd_generator_v2", ARENA / "generator.py")
generator = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(generator)


def _schema(path):
    return Draft202012Validator(json.loads((ARENA / path).read_text(encoding="utf-8")))


_ANSWER = {"deviations": [{"dimension": "sample_size", "deviation": False, "deviation_kind": None,
                           "confidence": 0.9, "registered_summary": None, "paper_summary": None}]}


def test_v2_schema_accepts_null_summaries_and_v1_still_rejects_them():
    assert not list(_schema("schemas/v2/output.schema.json").iter_errors(_ANSWER))
    # Two-sided control: v1 is frozen, so the same answer must still fail there.
    assert list(_schema("schemas/output.schema.json").iter_errors(_ANSWER))


def test_v1_ids_are_frozen_and_v2_ids_do_not_collide():
    v1 = [e["task_id"] for e in generator.generate("v1", 0)]
    v2 = [e["task_id"] for e in generator.generate("v2", 0)]
    assert v1[0] == "pd-t1-0-s0" and len(v1) == len(v2) == 46
    assert not set(v1) & set(v2)
    assert all(e["task_set_version"] == "v2" for e in generator.generate("v2", 0))


def test_v2_private_ids_carry_no_seed():
    a = [e["task_id"] for e in generator.generate("v2", 111111111, split="private")]
    b = [e["task_id"] for e in generator.generate("v2", 222222222, split="private")]
    assert a and a == b
    assert not any("111111111" in t for t in a)


def test_unknown_task_set_is_refused():
    import pytest
    with pytest.raises(ValueError):
        list(generator.generate("v3", 0))


def test_v2_prompt_does_not_invite_omitting_fields():
    """The first v2 wording ("use null (or omit them)") was followed by Opus 5 and
    Command R7B dropping `confidence` on 5/9 and 4/7 tasks (2026-10-08), against
    0/46 for Opus 5 on v1. The prompt must allow null for the summaries only."""
    text = (ARENA.parents[1] / "players" / "prompts" / "prereg_deviation_v2.txt").read_text(encoding="utf-8")
    assert "omit" not in text.lower()
    assert "including confidence" in text
