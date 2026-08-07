"""The Kaggle exporter must publish PUBLIC tasks only, and never nothing.

Export is publication. The two failure modes that matter are (a) a held-out task
reaching the exported file, and (b) an empty export that reads at the shell
exactly like a successful one.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from framework import export_kaggle
from framework.paths import ARENAS_ROOT_ENV

REPO = Path(__file__).resolve().parents[2]
ARENAS = REPO / "arenas"

pytestmark = pytest.mark.skipif(not ARENAS.is_dir(), reason="needs the arenas tree")

#: A seed-based arena with a small public split — cheap to generate in a test.
ARENA = "grim-consistency-v1"


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    out = tmp_path_factory.mktemp("kbench")
    n = export_kaggle.export_arena(ARENA, "v1", out)
    assert n > 0
    module = out / f"{ARENA.replace('-', '_')}.py"
    tasks = json.loads(module.with_suffix(".tasks.json").read_text(encoding="utf-8"))
    return module, tasks


def test_exports_a_nonzero_number_of_tasks(exported):
    _, tasks = exported
    assert tasks, "an empty export would look identical to a successful one"


def test_only_public_envelopes_are_exported(exported):
    """The contamination assertion. `is_held_out` treats a MISSING marker as
    held-out, so an unlabelled task is excluded rather than assumed safe."""
    _, tasks = exported
    visibilities = {t["envelope"].get("visibility") for t in tasks.values()}
    assert visibilities == {"public"}, (
        f"exported envelopes with visibility {visibilities} — anything other than "
        f"{{'public'}} means held-out or unmarked material was published"
    )


def test_the_generated_module_is_valid_python(exported):
    module, _ = exported
    ast.parse(module.read_text(encoding="utf-8"))


def test_the_generated_module_delegates_scoring(exported):
    """It must call the arena's scorer, not carry its own copy.

    A re-implemented scorer would make a Kaggle verdict silently mean something
    different from the published leaderboard number under the same arena name.
    """
    module, _ = exported
    src = module.read_text(encoding="utf-8")
    assert "_import_arena_module" in src and "_SCORER.score(" in src, (
        "the exported module does not delegate to the arena scorer"
    )


def test_refuses_to_write_an_empty_export(tmp_path, monkeypatch):
    """A held-out-only arena must raise, not emit a valid-looking empty file."""
    fake = tmp_path / "arenas" / "empty-v1"
    fake.mkdir(parents=True)
    (fake / "arena.yaml").write_text("arena_id: empty-v1\n", encoding="utf-8")

    monkeypatch.setenv(ARENAS_ROOT_ENV, str(tmp_path / "arenas"))
    monkeypatch.setattr(export_kaggle, "load_arena", lambda d: {"manifest": {}})
    monkeypatch.setattr(
        export_kaggle, "_import_arena_module",
        lambda d, name: type("M", (), {
            # Every envelope is held-out, so nothing may be exported.
            "generate": staticmethod(lambda ts, seed: [
                {"task_id": "t1", "visibility": "held_out", "input": {"text": "x"}},
            ]),
            "ground_truth": staticmethod(lambda tid: {"gold": "SECRET"}),
        })(),
    )

    with pytest.raises(SystemExit, match="REFUSING to write an empty export"):
        export_kaggle.export_arena("empty-v1", "v1", tmp_path / "out")

    assert not (tmp_path / "out").exists() or not list((tmp_path / "out").glob("*.py")), (
        "an empty export was written to disk despite the refusal"
    )


def test_an_unmarked_envelope_is_treated_as_held_out(tmp_path, monkeypatch):
    """The fail-safe direction, verified rather than assumed."""
    monkeypatch.setenv(ARENAS_ROOT_ENV, str(tmp_path))
    (tmp_path / "unmarked-v1").mkdir(parents=True)
    monkeypatch.setattr(export_kaggle, "load_arena", lambda d: {"manifest": {}})
    monkeypatch.setattr(
        export_kaggle, "_import_arena_module",
        lambda d, name: type("M", (), {
            # No `visibility` key at all — the exact shape H1 found in 136 records.
            "generate": staticmethod(lambda ts, seed: [
                {"task_id": "t1", "input": {"text": "x"}},
            ]),
            "ground_truth": staticmethod(lambda tid: {"gold": "SECRET"}),
        })(),
    )
    with pytest.raises(SystemExit, match="REFUSING"):
        export_kaggle.export_arena("unmarked-v1", "v1", tmp_path / "out")
