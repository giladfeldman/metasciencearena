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


def test_v6_private_ids_carry_nothing_derived_from_the_seed():
    """sha256(seed)[:8] over a 9-digit seed is inverted in under an hour (d-c7642f)."""
    a = generator._seed_tag("v6", SEED, "private")
    b = generator._seed_tag("v6", SEED + 1, "private")
    assert a == b, "the private tag still depends on the seed"
    assert str(SEED) not in a
    assert generator._seed_tag("v7", SEED, "private") != a
    # Frozen versions keep their committed private ids.
    import hashlib
    assert generator._seed_tag("v5", SEED, "private") == \
        hashlib.sha256(str(SEED).encode("utf-8")).hexdigest()[:8]


def test_v6_generated_private_ids_are_seed_free_and_unique():
    generator._GROUND_TRUTH_CACHE.clear()
    ids_a = [e["task_id"] for e in generator.generate("v6", 111111111, split="private")]
    generator._GROUND_TRUTH_CACHE.clear()
    ids_b = [e["task_id"] for e in generator.generate("v6", 222222222, split="private")]
    generator._GROUND_TRUTH_CACHE.clear()
    rev = {e["task_id"] for e in generator.generate("v6", 0, split="revealed")}
    generator._GROUND_TRUTH_CACHE.clear()
    assert ids_a, "generator produced no tasks -- the probe is broken"
    assert ids_a == ids_b, "private ids still change with the seed"
    assert len(set(ids_a)) == len(ids_a)
    assert rev.isdisjoint(ids_a)
