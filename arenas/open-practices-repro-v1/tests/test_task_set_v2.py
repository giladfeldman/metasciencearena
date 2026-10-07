"""v2 exists to stop generating tasks around a real personal directory.

The string itself is NOT written here. This file is published to the public
mirror (the arena's tests are listed in contract/public_manifest.txt), so a test
asserting on the literal would re-publish the very thing v2 exists to stop
using, and would keep republishing it long after v1 stops being served. The
suspect string is instead DERIVED at run time as the one entry of the v1 pool
with no counterpart in the v2 pool, which also makes the test self-maintaining:
it cannot drift away from what v1 actually contains.

Two-sided throughout. Every "v2 does not contain it" assertion is paired with a
positive control proving the locator finds it where it IS -- the failure mode
these guard against is a search that returns zero because it is broken.
"""
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pytest
import yaml

ARENA_DIR = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "_open_practices_repro_generator_v2", ARENA_DIR / "generator.py")
generator = importlib.util.module_from_spec(_SPEC)
sys.modules["_open_practices_repro_generator_v2"] = generator
_SPEC.loader.exec_module(generator)

# Through the generator's own loader, so `abs_dirs_v1` (the private v1 pool, see
# repo_templates.yaml) is folded in exactly when the generator would use it.
CATALOG = generator._load_catalog()
PUBLIC_CATALOG = yaml.safe_load(
    (ARENA_DIR / "catalogs" / "repo_templates.yaml").read_text(encoding="utf-8"))
HAVE_V1_AS_PUBLISHED = generator.V1_AS_PUBLISHED.is_file()
needs_v1_as_published = pytest.mark.skipif(
    not HAVE_V1_AS_PUBLISHED,
    reason="abs_dirs_v1_as_published.yaml is private; the public mirror cannot "
           "regenerate the published v1 byte-for-byte")
PRIVATE_SEED = 12345   # any private-split seed; the point is the split, not the value


def _tasks(version, seed, split):
    """{task_id: (envelope, gold)} for one (version, seed, split)."""
    out = {}
    for env in generator.generate(version, seed, split=split):
        out[env["task_id"]] = (env, generator.ground_truth(env["task_id"]))
    return out


def _count(tasks, needle):
    return sum(1 for pair in tasks.values() if needle in json.dumps(pair, sort_keys=True))


@pytest.fixture(scope="module")
def replaced_string():
    """The v1 pool entry v2 replaced, derived rather than written down.

    Asserting there is EXACTLY one difference each way is itself the guard: if a
    later edit changed two entries, or none, this fails rather than silently
    testing nothing.
    """
    v1_pool = generator._abs_dirs_for("v1", CATALOG)
    v2_pool = generator._abs_dirs_for("v2", CATALOG)
    gone = [d for d in v1_pool if d not in v2_pool]
    added = [d for d in v2_pool if d not in v1_pool]
    assert len(gone) == 1, gone
    assert len(added) == 1, added
    return gone[0], added[0]


def test_the_pools_are_the_same_length():
    """rng.choice spends one draw either way, but a different length changes
    which element each draw lands on -- that is task SHAPE, not just a string."""
    assert len(CATALOG["abs_dirs_v2"]) == len(CATALOG["abs_dirs"])


def test_abs_dirs_for_refuses_a_pool_of_a_different_length():
    bad = dict(CATALOG, abs_dirs_v9=CATALOG["abs_dirs"][:-1])
    with pytest.raises(ValueError, match="task shape"):
        generator._abs_dirs_for("v9", bad)


def test_an_undeclared_version_falls_back_to_the_base_pool():
    assert generator._abs_dirs_for("v99", CATALOG) == CATALOG["abs_dirs"]
    assert generator._abs_dirs_for("v2", CATALOG) == CATALOG["abs_dirs_v2"]


def test_v1_still_renders_the_replaced_string_positive_control(replaced_string):
    """THE CONTROL. If this goes to zero the locator is broken, or v1 drifted --
    either way every assertion below is meaningless and must not read as green."""
    gone, _ = replaced_string
    assert _count(_tasks("v1", 0, "revealed"), gone) > 0
    assert _count(_tasks("v1", PRIVATE_SEED, "private"), gone) > 0


def test_v2_renders_the_replaced_string_nowhere(replaced_string):
    gone, _ = replaced_string
    assert _count(_tasks("v2", 0, "revealed"), gone) == 0
    assert _count(_tasks("v2", PRIVATE_SEED, "private"), gone) == 0


def test_v2_renders_the_replacement_positive_control(replaced_string):
    """Proves v2 draws from its own pool at all. Without this, a v2 that had
    stopped rendering absolute paths entirely would pass the test above."""
    _, added = replaced_string
    assert _count(_tasks("v2", 0, "revealed"), added) > 0
    assert _count(_tasks("v2", PRIVATE_SEED, "private"), added) > 0


@needs_v1_as_published
def test_v2_committed_gold_does_not_carry_the_replaced_string(replaced_string):
    """The committed gold is what build-data.mjs publishes, so it is the file
    that actually reaches the site."""
    gone, added = replaced_string
    v2 = (ARENA_DIR / "task_sets" / "v2" / "_ground_truth.json").read_text(encoding="utf-8")
    v1 = (ARENA_DIR / "task_sets" / "v1" / "_ground_truth.json").read_text(encoding="utf-8")
    assert gone not in v2
    assert added in v2                    # positive control: v2 gold is real gold
    assert gone in v1                     # positive control: the search works


@needs_v1_as_published
def test_v1_committed_gold_reproduces_byte_for_byte():
    """v1 is superseded and its published scores were computed on these exact
    tasks. Minting v2 must not have moved a byte of it."""
    committed = (ARENA_DIR / "task_sets" / "v1" / "_ground_truth.json").read_text(encoding="utf-8")
    regen = {}
    for env in generator.generate("v1", 0, split="revealed"):
        regen[env["task_id"]] = {"envelope": env,
                                 "ground_truth": generator.ground_truth(env["task_id"])}
    assert json.dumps(regen, indent=2, sort_keys=True) == committed


def _shape(tasks):
    return {
        "task_ids": sorted(tasks),
        "tiers": Counter(e["difficulty"]["tier"] for e, _ in tasks.values()),
        "n_issues": Counter(e["difficulty"]["n_issues"] for e, _ in tasks.values()),
        "kinds": Counter(k for _, g in tasks.values() for k in (g["mistake_kinds"] or ["clean"])),
        "n_files": Counter(len(e["input"]["files"]) for e, _ in tasks.values()),
        "n_gold": Counter(len(g["records"]) for _, g in tasks.values()),
    }


@pytest.mark.parametrize("seed,split", [(0, "revealed"), (PRIVATE_SEED, "private")])
def test_v2_has_the_same_task_shape_as_v1(seed, split):
    """v2 is a different STRING, not a different task design."""
    assert _shape(_tasks("v1", seed, split)) == _shape(_tasks("v2", seed, split))


@pytest.mark.parametrize("seed,split", [(0, "revealed"), (PRIVATE_SEED, "private")])
def test_v2_is_a_fresh_draw_not_a_re_render(seed, split):
    """_seed_int hashes the version in, so v2 is not comparable with v1. If this
    ever went to zero differences, comparable_with_previous: false in arena.yaml
    would be a false statement."""
    def texts(v):
        return {tid: e["input"]["repo_url"] + "".join(f["content"] for f in e["input"]["files"])
                for tid, (e, _) in _tasks(v, seed, split).items()}
    a, b = texts("v1"), texts("v2")
    same = [tid for tid in a if a[tid] == b[tid]]
    assert len(same) <= 1, same     # or-t5-1-s0 collides on a 4-way repo_url draw


def test_envelopes_carry_the_version_they_were_generated_as():
    """It was hardcoded to "v1" before v2 existed, which would have stamped every
    v2 record with the wrong task set and made them invisible to the build."""
    for v in ("v1", "v2"):
        versions = {e["task_set_version"] for e in generator.generate(v, 0, split="revealed")}
        assert versions == {v}


def test_this_test_file_does_not_itself_publish_the_string(replaced_string):
    """This file is in the public manifest. Writing the literal here would
    re-publish it, which is the failure this whole change exists to stop."""
    gone, _ = replaced_string
    source = Path(__file__).read_text(encoding="utf-8")
    assert gone not in source
    assert "abs_dirs_v2" in source        # positive control: the check can find one


def test_arena_yaml_declares_v2_current_and_v1_superseded():
    manifest = yaml.safe_load((ARENA_DIR / "arena.yaml").read_text(encoding="utf-8"))
    ladder = {v["version"]: v for v in manifest["task_set_versions"]}
    assert ladder["v1"]["status"] == "superseded"
    assert ladder["v1"]["superseded_by"] == "v2"
    assert ladder["v2"]["status"] == "current"
    assert ladder["v2"]["supersedes"] == "v1"
    assert ladder["v2"]["comparable_with_previous"] is False
    for v in ("v1", "v2"):
        assert re.fullmatch(r"sha256:[0-9a-f]{64}", ladder[v]["generator_fingerprint"])


def test_the_public_catalog_does_not_carry_the_replaced_string(replaced_string):
    """The catalog is mirrored. The v1 string lives only in the private file."""
    gone, _ = replaced_string
    public = (ARENA_DIR / "catalogs" / "repo_templates.yaml").read_text(encoding="utf-8")
    if HAVE_V1_AS_PUBLISHED:
        assert gone not in public
        assert gone in generator.V1_AS_PUBLISHED.read_text(encoding="utf-8")  # control
    assert "abs_dirs_v2" in public          # positive control: the right file was read


def test_the_public_v1_placeholder_has_the_published_shape():
    """Same length and same entry kinds as the private pool, so the public v1
    draws the same task shape -- only the rendered string differs."""
    public_v1 = PUBLIC_CATALOG["abs_dirs"]
    assert len(public_v1) == len(generator._abs_dirs_for("v1", CATALOG))
    if HAVE_V1_AS_PUBLISHED:
        private_v1 = CATALOG["abs_dirs_v1"]
        diff = [i for i, (a, b) in enumerate(zip(public_v1, private_v1)) if a != b]
        assert len(diff) == 1, diff       # exactly the one personal entry, nothing else
        assert public_v1[diff[0]].startswith("C:/Users/")
        assert public_v1[diff[0]].endswith("/Dropbox/thesis/raw")


def test_v1_without_the_private_pool_warns_rather_than_passing_as_published(monkeypatch, tmp_path):
    monkeypatch.setattr(generator, "V1_AS_PUBLISHED", tmp_path / "absent.yaml")
    with pytest.warns(UserWarning, match="input hashes do not"):
        list(generator.generate("v1", 0, split="revealed"))


@needs_v1_as_published
def test_v1_with_the_private_pool_does_not_warn():
    """The other side: the private repo regenerates the real v1 silently."""
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("error")
        list(generator.generate("v1", 0, split="revealed"))

