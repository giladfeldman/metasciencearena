"""The Inspect AI exporter must publish PUBLIC tasks only, and never nothing.

Same contract as the Kaggle exporter (see test_export_kaggle.py), and the same
two failure modes: (a) a held-out task reaching the exported file, and (b) an
empty export that reads at the shell exactly like a successful one.

This exporter matters a little more than the Kaggle one for contamination,
because Inspect is the format the public benchmarking hubs (OpenBench, Epoch AI)
ingest — an export that leaked would not sit in a private notebook, it would be
mirrored by third parties.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from framework import export_inspect
from framework.holdout import is_held_out
from framework.paths import ARENAS_ROOT_ENV

REPO = Path(__file__).resolve().parents[2]
ARENAS = REPO / "arenas"

pytestmark = pytest.mark.skipif(not ARENAS.is_dir(), reason="needs the arenas tree")

#: A seed-based arena with a small public split — cheap to generate in a test.
ARENA = "grim-consistency-v1"


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    out = tmp_path_factory.mktemp("inspect")
    n = export_inspect.export_arena(ARENA, "v1", out)
    assert n > 0
    module = out / f"{ARENA.replace('-', '_')}.py"
    tasks = json.loads(module.with_suffix(".tasks.json").read_text(encoding="utf-8"))
    return module, tasks


def test_exports_a_nonzero_number_of_tasks(exported):
    _, tasks = exported
    assert tasks, "an empty export would look identical to a successful one"


def test_only_public_envelopes_are_exported(exported):
    """The contamination invariant, checked through the same fail-safe predicate
    the rest of the framework uses rather than by reading the marker directly."""
    _, tasks = exported
    leaked = [tid for tid, t in tasks.items() if is_held_out(t["envelope"])]
    assert leaked == [], f"held-out tasks reached an Inspect export: {leaked[:5]}"


def test_the_generated_module_is_valid_python(exported):
    module, _ = exported
    ast.parse(module.read_text(encoding="utf-8"))


def test_the_generated_module_delegates_scoring(exported):
    """It must call the arena's scorer, not carry its own copy.

    A re-implemented scorer would make an OpenBench/Epoch number silently mean
    something different from the published leaderboard under the same arena name
    — and unlike a local mistake, that one propagates to other people's sites.

    NOTE: this is a source assertion and therefore WEAK — it proves the text
    mentions the scorer, not that the number matches. The real guarantee is
    test_exported_scorer_returns_the_same_number_as_the_arena_scorer below, which
    executes it. Until 2026-08-15 only this substring check existed, `inspect_ai`
    was not a dependency, and the exported module had never once been imported —
    while the funder letters claimed exact leaderboard equivalence.
    """
    module, _ = exported
    src = module.read_text(encoding="utf-8")
    assert "_import_arena_module" in src and "_SCORER.score(" in src, (
        "the exported module does not delegate to the arena scorer"
    )


def _load_exported(module_path: Path):
    """Import the generated module (needs inspect_ai + the arenas root)."""
    pytest.importorskip("inspect_ai", reason="pip install 'metasciencearena[inspect]'")
    import importlib.util

    spec = importlib.util.spec_from_file_location("exported_arena_task", module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _State:
    """Minimal stand-in for Inspect's TaskState — the two fields the scorer reads."""

    def __init__(self, task_id: str, completion: str):
        self.metadata = {"task_id": task_id}
        self.output = type("O", (), {"completion": completion})()


def _valid_output(envelope: dict) -> str:
    return json.dumps({"records": [
        {"stat_id": s["stat_id"], "flagged": False, "issue_kind": None, "confidence": 0.9}
        for s in envelope["input"]["statistics"]
    ]})


def test_exported_scorer_returns_the_same_number_as_the_arena_scorer(exported):
    """The equivalence claim the outreach letters make, actually executed.

    "A number produced in Inspect means exactly what the number on our
    leaderboard means" is a claim about behaviour; the only way to hold it is to
    run both scorers on the same output and compare.
    """
    import asyncio

    module, tasks = exported
    mod = _load_exported(module)
    task_id = sorted(tasks)[0]
    entry = tasks[task_id]
    raw = _valid_output(entry["envelope"])

    direct = mod._SCORER.score(json.loads(raw), entry["ground_truth"])["primary"]
    via_inspect = asyncio.run(mod.arena_scorer()(_State(task_id, raw), None)).value
    assert float(via_inspect) == float(direct), (
        f"exported scorer {via_inspect} != arena scorer {direct} for the same output"
    )


@pytest.mark.parametrize("label,bad", [
    ("unparseable", "I refuse to answer."),
    ("schema_invalid", '{"records": [{"stat_id": "x"}]}'),
])
def test_rejected_output_raises_instead_of_scoring_zero(exported, label, bad):
    """A task the leaderboard EXCLUDES must not be scored 0.0 here.

    The runner records unparseable/schema-invalid output with `breakdown.error`
    (runner.py:648-656) and both report engines drop errored records from the
    mean. Inspect's `accuracy()` is a mean over every sample it is given, so a
    0.0 would drag the published number down: measured on 2026-08-15, 27 answers
    at 0.9 plus 3 malformed gives 0.90 on the leaderboard and 0.81 under
    accuracy(). Raising makes Inspect record a sample error, which it excludes.
    """
    import asyncio

    module, tasks = exported
    mod = _load_exported(module)
    task_id = sorted(tasks)[0]
    with pytest.raises(mod.ArenaOutputRejected):
        asyncio.run(mod.arena_scorer()(_State(task_id, bad), None))


def test_inspect_accuracy_would_have_diverged_without_the_exclusion():
    """Pins the MEASUREMENT behind the fix above, so nobody re-litigates it.

    If a future Inspect release makes accuracy() skip zeros, this goes red and
    the exclusion can be revisited — rather than the rationale silently rotting.
    """
    pytest.importorskip("inspect_ai", reason="pip install 'metasciencearena[inspect]'")
    from inspect_ai.scorer import Score, SampleScore, accuracy

    def s(v):
        return SampleScore(score=Score(value=v))

    good = [s(0.9)] * 27
    assert accuracy()(good) == pytest.approx(0.90), "leaderboard semantics: errored excluded"
    assert accuracy()(good + [s(0.0)] * 3) == pytest.approx(0.81), (
        "accuracy() averages every sample — scoring rejects as 0.0 would diverge"
    )


def test_refuses_to_write_an_empty_export(tmp_path, monkeypatch):
    """A held-out-only arena must raise, not emit a valid-looking empty file."""
    fake = tmp_path / "arenas" / "empty-v1"
    fake.mkdir(parents=True)
    (fake / "arena.yaml").write_text("arena_id: empty-v1\n", encoding="utf-8")

    monkeypatch.setenv(ARENAS_ROOT_ENV, str(tmp_path / "arenas"))
    monkeypatch.setattr(export_inspect, "load_arena", lambda d: {"manifest": {}})
    monkeypatch.setattr(
        export_inspect, "_import_arena_module",
        lambda d, name: type("M", (), {
            "generate": staticmethod(lambda ts, seed: [
                {"task_id": "t1", "visibility": "held_out", "input": {"text": "x"}},
            ]),
            "ground_truth": staticmethod(lambda tid: {"gold": "SECRET"}),
        })(),
    )

    with pytest.raises(SystemExit, match="REFUSING to write an empty export"):
        export_inspect.export_arena("empty-v1", "v1", tmp_path / "out")

    assert not (tmp_path / "out").exists() or not list((tmp_path / "out").glob("*.py")), (
        "an empty export was written to disk despite the refusal"
    )


def test_an_unmarked_envelope_is_treated_as_held_out(tmp_path, monkeypatch):
    """The fail-safe direction, verified rather than assumed."""
    monkeypatch.setenv(ARENAS_ROOT_ENV, str(tmp_path))
    (tmp_path / "unmarked-v1").mkdir(parents=True)
    monkeypatch.setattr(export_inspect, "load_arena", lambda d: {"manifest": {}})
    monkeypatch.setattr(
        export_inspect, "_import_arena_module",
        lambda d, name: type("M", (), {
            # No `visibility` key at all.
            "generate": staticmethod(lambda ts, seed: [
                {"task_id": "t1", "input": {"text": "x"}},
            ]),
            "ground_truth": staticmethod(lambda tid: {"gold": "SECRET"}),
        })(),
    )
    with pytest.raises(SystemExit, match="REFUSING"):
        export_inspect.export_arena("unmarked-v1", "v1", tmp_path / "out")


def test_gold_never_appears_in_the_module_source(exported):
    """Gold lives in the .tasks.json sidecar (public gold, by design) and must not
    be baked into the emitted module, which is the file people paste around."""
    module, tasks = exported
    src = module.read_text(encoding="utf-8")
    assert "ground_truth" not in src.split('"""', 2)[-1] or "_TASKS[task_id]" in src, (
        "the module appears to inline ground truth rather than read the sidecar"
    )
