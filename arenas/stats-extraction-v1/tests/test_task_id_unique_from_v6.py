"""Task ids must not be reused across task-set versions from v6 on (debt d-dc055d).

v2..v5 hash the seed alone, so they share all 36 revealed ids with different text.
They are frozen, so their ids must NOT move (stored records and links key off them).
From v6 the version is hashed in, so a new task set never reuses an old id.
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "se_generator_ids", Path(__file__).resolve().parent.parent / "generator.py")
generator = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(generator)

SEED = 123456789


def test_frozen_versions_keep_their_legacy_tags():
    import hashlib
    legacy = hashlib.sha256(str(SEED).encode("utf-8")).hexdigest()[:8]
    assert generator._seed_tag("v1", SEED) == str(SEED)
    for v in ("v2", "v3", "v4", "v5"):
        assert generator._seed_tag(v, SEED) == legacy, v


def test_v6_tag_differs_from_every_earlier_version_and_from_v7():
    earlier = {generator._seed_tag(v, SEED) for v in ("v1", "v2", "v3", "v4", "v5")}
    v6, v7 = generator._seed_tag("v6", SEED), generator._seed_tag("v7", SEED)
    assert v6 not in earlier
    assert v7 not in earlier and v7 != v6
    assert len(v6) == 8


def test_v6_generated_ids_do_not_collide_with_v5():
    generator._GROUND_TRUTH_CACHE.clear()
    v5 = {e["task_id"] for e in generator.generate("v5", 0, split="revealed")}
    generator._GROUND_TRUTH_CACHE.clear()
    v6 = {e["task_id"] for e in generator.generate("v6", 0, split="revealed")}
    generator._GROUND_TRUTH_CACHE.clear()
    assert v5 and v6, "generator produced no tasks -- the probe is broken"
    assert len(v6) == len(v5)
    assert v5.isdisjoint(v6)
