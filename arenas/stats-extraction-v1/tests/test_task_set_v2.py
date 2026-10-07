"""Task-set v2 must not re-leak the seed it was created to rotate.

v1's private seed is recoverable from tracked data because the generator builds
`t-tier{tier}-d{density}-{k}-s{seed}` and private run records are committed:
`runs/v1/escimate__private__v0_6_13.jsonl` carries the secret in 36 `task_id`s.
Rotating to a fresh seed fixes nothing on its own — v2's records would embed the
NEW seed in exactly the same way, and the arena would be back where it started
after one tournament.

So v2 changes the *format*, not just the value. v1's ids are left byte-identical
(36 stored records and every `task_id`-keyed consumer — build-data.mjs, the
task-detail route, compare-task-diff, player-report — depend on them), while v2+
derives the discriminator from a truncated SHA-256 of the seed: stable and
deterministic, but not invertible.

Also pinned here: the envelope's `task_set_version`. Four `_build_*` helpers
hardcoded `"v1"`, so v2 tasks — and therefore v2 run records — would have
announced themselves as v1, which is the same mislabel
`leaderboard-app/lib/__tests__/task-set-version-agreement.test.ts` catches on the
published side.
"""
import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

try:
    from framework.tests._corpora import requires_corpora
except ImportError:  # pragma: no cover - only in the public mirror
    # `_corpora` reaches the answer-key route and is never mirrored, but a
    # MODULE-LEVEL import of it killed pytest COLLECTION in the public repo —
    # so the six seed-format assertions below, which need no corpus at all, did
    # not run for anyone who cloned it. Guarding the import keeps them running.
    #
    # The fallback is a SKIP, never a no-op decorator: "did not run" must not be
    # able to read as "passed". `test_publish.py::
    # test_the_corpora_fallback_skips_rather_than_silently_passing` pins that.
    requires_corpora = pytest.mark.skip(
        reason="framework.tests._corpora is private (it resolves the gold corpus); "
               "this assertion needs the local-only corpora and cannot run here"
    )

ARENA_DIR = Path(__file__).resolve().parents[1]

if str(ARENA_DIR) not in sys.path:
    sys.path.insert(0, str(ARENA_DIR))
_spec = importlib.util.spec_from_file_location("_gen_v2", ARENA_DIR / "generator.py")
generator = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(generator)

#: A DUMMY 9-digit seed. It must never be a real one: this file is tracked, and
#: `framework.publish.scan_for_leaks` greps tracked files for every live
#: `.private_seed` value. The first draft of this module pasted the actual v1 seed
#: out of the handoff and was flagged immediately — the same mistake, and the same
#: catch, that `framework/tests/test_private_seed_not_in_tracked_files.py` records
#: in its own docstring. The assertion below keeps it that way.
SECRET = 123456789


@requires_corpora
def test_the_dummy_seed_is_not_a_real_one():
    """Guards this file against becoming the leak it was written to prevent."""
    live = {
        p.read_text(encoding="utf-8").strip()
        for p in ARENA_DIR.parent.glob("*/task_sets/*/.private_seed")
    }
    assert live, "no .private_seed found — this check would be vacuous"
    assert str(SECRET) not in live, "SECRET is a LIVE private seed; pick an unused value"


def test_v1_task_ids_keep_the_legacy_format():
    """Backward compatibility is the constraint that forced a new task set."""
    envs = list(generator.generate("v1", SECRET, "private"))
    assert envs, "v1 generated no tasks"
    assert all(e["task_id"].endswith(f"-s{SECRET}") for e in envs), (
        "v1 task_ids must stay byte-identical — 36 stored records key off them"
    )


def test_v2_task_ids_never_contain_the_raw_seed():
    """The whole point of the rotation."""
    envs = list(generator.generate("v2", SECRET, "private"))
    assert envs, "v2 generated no tasks"
    leaked = [e["task_id"] for e in envs if str(SECRET) in e["task_id"]]
    assert leaked == [], f"v2 task_ids still embed the secret seed: {leaked[:3]}"


def test_v2_task_ids_use_the_hashed_discriminator_and_are_stable():
    digest = hashlib.sha256(str(SECRET).encode()).hexdigest()[:8]
    envs = list(generator.generate("v2", SECRET, "private"))
    assert all(e["task_id"].endswith(f"-s{digest}") for e in envs)
    again = list(generator.generate("v2", SECRET, "private"))
    assert [e["task_id"] for e in envs] == [e["task_id"] for e in again], "not deterministic"


def test_v2_task_ids_are_unique_and_distinguish_seeds():
    a = [e["task_id"] for e in generator.generate("v2", SECRET, "private")]
    b = [e["task_id"] for e in generator.generate("v2", SECRET + 1, "private")]
    assert len(set(a)) == len(a), "duplicate task_ids within one split"
    assert set(a).isdisjoint(b), "different seeds must not collide"


@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.parametrize("split", ["revealed", "private"])
def test_envelopes_report_the_task_set_version_they_were_generated_for(version, split):
    seed = 0 if split == "revealed" else SECRET
    envs = list(generator.generate(version, seed, split))
    assert envs, f"{version}/{split} generated no tasks"
    wrong = {e["task_set_version"] for e in envs} - {version}
    assert not wrong, (
        f"{version}/{split}: envelopes claim task_set_version {wrong} — a run record "
        "written from these would mislabel which task set produced the score"
    )


def test_sub_task_ids_of_tier6_also_avoid_the_seed():
    """Tier 6 composes child ids as f'{task_id}-sub{i}'; the parent must be safe."""
    envs = list(generator.generate("v2", SECRET, "private"))
    subs = [e["task_id"] for e in envs if "-sub" in e["task_id"]]
    assert all(str(SECRET) not in s for s in subs), subs[:3]
