"""The declared-instrument identity layer, and the ways it could be wrong.

WHAT IT REPLACED, AND WHY. ``/players`` used to key on the normalised
``player_version``. That string labels WEIGHTS and records no inference
configuration, so the same GGUF run with thinking on and with thinking off
declares one identical string and renders as ONE card -- while differing by
+0.24 to +0.28 mean score, the largest single effect measured in this corpus.
Three consult seats (2026-09-16) converged on a DECLARED instrument: one
hand-written line per registry entry, audited against the settings the records
say were actually sent.

THE FIXTURES ARE SHARED WITH NODE. Every pair below comes from
``framework/contract/fixtures/player_identity_cases.json``, which
``leaderboard-app/lib/__tests__/model-family.test.ts`` reads too. Both engines
compute published numbers independently, and a design implemented in one and not
the other has shipped a page disagreeing with a report on this repo before.

THE PAIRS ARE PINNED AS STRINGS, NOT READ OFF THE BUNDLE. A bundle-only test is
how the ``familySlug`` collision stayed green: the bundle cannot contain the
offending pair until the bad url has already shipped.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

from framework import identity
from framework.registry import undeclared_instruments

REPO = Path(__file__).resolve().parents[2]
FIXTURES = json.loads(
    (REPO / "framework" / "contract" / "fixtures" / "player_identity_cases.json")
    .read_text(encoding="utf-8"))
REGISTRY = yaml.safe_load((REPO / "players" / "registry.yaml").read_text(encoding="utf-8"))


# ------------------------------------------------------------------ the fixtures

def test_the_fixture_corpus_is_big_enough_for_the_assertions_below_to_mean_anything():
    """A green computed from an empty input is a false green.

    Every parametrised test below passes trivially on zero cases, and both
    halves have to be present: a key that merges everything and a key that
    merges nothing each pass a one-sided corpus.
    """
    cases = FIXTURES["cases"]
    assert len(cases) >= 12
    assert sum(1 for c in cases if c["same_family"]) >= 3, "no known-EQUAL pairs"
    assert sum(1 for c in cases if not c["same_family"]) >= 8, "no known-DIFFERENT pairs"
    assert len(FIXTURES["inference_config"]["cases"]) >= 5
    assert len(FIXTURES["slug_identity"]["instruments"]) >= 5


@pytest.mark.parametrize("case", FIXTURES["cases"], ids=lambda c: c["name"])
def test_declared_instruments_group_exactly_as_the_fixture_says(case):
    """The instrument IS the family key. One line per pair, both directions."""
    a, b = case["a"]["instrument"], case["b"]["instrument"]
    if case["same_family"]:
        assert a == b, case["why"]
    else:
        assert a != b, case["why"]


@pytest.mark.parametrize("case", FIXTURES["cases"], ids=lambda c: c["name"])
def test_declared_instruments_never_collide_on_their_slug(case):
    """The url half, which is where the real defect was.

    ``familySlug`` squeezes every non-alphanumeric RUN to one hyphen, so ``/``
    and ``-`` become the same character and two players can share a url while
    their keys differ. Declared instruments are already in that output shape, so
    the slug is the identity on them -- asserted here rather than assumed.
    """
    a, b = case["a"]["instrument"], case["b"]["instrument"]
    same_slug = _slug(a) == _slug(b) if (a and b) else (a == b)
    assert same_slug == case["same_slug"], case["why"]


def _slug(key: str | None) -> str | None:
    """The Python twin of leaderboard-app/lib/model-family.ts familySlug."""
    if key is None:
        return None
    return re.sub(r"^-+|-+$", "", re.sub(r"[^a-z0-9]+", "-", key.lower()))


@pytest.mark.parametrize("inst", FIXTURES["slug_identity"]["instruments"])
def test_slugging_a_declared_instrument_returns_it_unchanged(inst):
    """This identity is what makes the collision unconstructible rather than untested."""
    assert _slug(inst) == inst
    assert identity.INSTRUMENT_RE.match(inst)


def test_the_glm_pair_really_does_collide_under_the_old_key():
    """TWO-SIDED CONTROL for the test above.

    If the collision cannot be reproduced here, the passing assertions above
    prove nothing -- a slug function that never collides would satisfy them
    while the real one still served one arm's page at the other's url. So
    reproduce it, with a negative control alongside.
    """
    a = "local-gguf/GLM-4.7-Flash/UD-Q4_K_XL"
    b = "local-gguf/GLM-4.7-Flash-UD-Q4_K_XL"
    assert a != b, "the two declared versions are distinct strings"
    assert _slug(a) == _slug(b) == "local-gguf-glm-4-7-flash-ud-q4-k-xl"
    # Negative control: two genuinely different models must NOT collide, or the
    # comparison above is reporting a broken slug rather than a real collision.
    assert _slug("groq/openai/gpt-oss-120b") != _slug("openrouter/openai/gpt-oss-120b")
    assert _slug("local-gguf/gemma-4-26B-A4B-it-UD-Q4_K_M") != _slug("local-gguf/gemma-4-26B-it/QAT-Q4_0")


# ------------------------------------------------------- canonical config + hash

@pytest.mark.parametrize("case", FIXTURES["inference_config"]["cases"], ids=lambda c: c["name"])
def test_inference_config_canonicalises_as_the_fixture_says(case):
    got = identity.inference_config(case["entry"])
    assert got == case["expect"], case.get("why", case["name"])
    sha = identity.inference_config_sha256(got)
    assert (sha is None) == case["expect_sha_null"]
    if sha is not None:
        assert re.fullmatch(r"[0-9a-f]{64}", sha)


def test_a_nested_block_hashes_the_same_whatever_order_it_was_written_in():
    """The one place Python and Node could silently disagree.

    ``json.dumps(sort_keys=True)`` recurses; ``JSON.stringify`` does not sort at
    all, so the Node mirror sorts by hand. A divergence here would report a split
    that does not exist -- two entries of one instrument, flagged as two.
    """
    one = {"openai_extra_body": {"chat_template_kwargs": {"enable_thinking": False, "a": 1}}}
    two = {"openai_extra_body": {"chat_template_kwargs": {"a": 1, "enable_thinking": False}}}
    assert (identity.inference_config_sha256(identity.inference_config(one))
            == identity.inference_config_sha256(identity.inference_config(two)))


def test_a_config_that_differs_really_does_hash_differently():
    """Two-sided control for the test above: an equality test alone would pass
    on a hash function that returned a constant."""
    low = {"openai_temperature": 0.0, "openai_extra_body": {"reasoning_effort": "low"}}
    high = {"openai_temperature": 0.0, "openai_extra_body": {"reasoning_effort": "high"}}
    assert (identity.inference_config_sha256(identity.inference_config(low))
            != identity.inference_config_sha256(identity.inference_config(high)))


def test_the_absence_of_a_config_is_not_the_hash_of_the_word_null():
    """`None` in, `None` out. Hashing the absence would make 'this player sends
    no settings' indistinguishable from a recorded settings block, and the whole
    point of the absent case is that it is a different answer."""
    assert identity.inference_config_sha256(None) is None


# ------------------------------------------------------------- the real registry

def test_every_entry_in_the_real_registry_declares_an_instrument():
    assert len(REGISTRY) >= 250, "two-sided: the registry really loaded"
    assert undeclared_instruments(REGISTRY) == []


def test_the_real_registry_passes_its_own_audit():
    assert identity.audit_registry(REGISTRY) == []


def test_the_audit_is_not_vacuous():
    """THE CONTROL THAT MAKES THE TEST ABOVE WORTH ANYTHING.

    An audit that returned [] unconditionally would pass every assertion in this
    file. Break each of the three rules on a COPY of the real registry and
    require it to fire.
    """
    import copy

    # Rule A: one instrument, two different inference configs.
    a = copy.deepcopy(REGISTRY)
    hits = [e for e in a if e.get("instrument") == "local-gguf-gpt-oss-20b-q8-rlow"]
    assert len(hits) >= 2, "two-sided: the fixture player really is in the registry"
    hits[0]["openai_extra_body"] = dict(hits[0]["openai_extra_body"], reasoning_effort="high")
    assert any("DIFFERENT inference configs" in p for p in identity.audit_registry(a))

    # Rule B: one identical shape declared under two names.
    b = copy.deepcopy(REGISTRY)
    same = [e for e in b if e.get("instrument") == "claude-opus-5"
            and e.get("adapter_class") == "SubprocessCliAdapter"]
    assert len(same) >= 2
    same[0]["instrument"] = "claude-opus-5-alt"
    assert any("different instruments" in p for p in identity.audit_registry(b))

    # Rule C: a name that is not slug-safe.
    c = copy.deepcopy(REGISTRY)
    c[0]["instrument"] = "local-gguf/Bad_Name"
    assert any("not slug-safe" in p for p in identity.audit_registry(c))


def test_the_registry_loader_refuses_an_inconsistent_declaration(tmp_path):
    """The rules are enforced at LOAD, not only in a test that someone must run."""
    from framework.registry import RegistryError, load_registry

    base = {
        "player_type": "ai-model", "player_version": "local-gguf/x-Q8_0",
        "adapter_class": "LocalGgufAdapter", "confidence_strategy": "native",
        "deterministic": False, "instrument": "local-x-q8",
    }
    good = [dict(base, player_id="x-grim"), dict(base, player_id="x-pubbias")]
    p = tmp_path / "registry.yaml"
    p.write_text(yaml.safe_dump(good), encoding="utf-8")
    assert len(load_registry(p)) == 2       # two-sided: the good case really loads

    bad = [dict(base, player_id="x-grim"),
           dict(base, player_id="x-pubbias",
                openai_extra_body={"reasoning_effort": "high"})]
    p.write_text(yaml.safe_dump(bad), encoding="utf-8")
    with pytest.raises(RegistryError, match="DIFFERENT inference configs"):
        load_registry(p)


# ------------------------------------------------------------- the record stamp

def test_the_runner_stamps_player_identity_on_every_record_it_writes():
    """A field nothing writes is a declaration, not a capability.

    Greps the real call site rather than a copy. The end-to-end proof that the
    block reaches disk is test_the_runner_really_writes_the_block below.
    """
    src = (REPO / "framework" / "runner.py").read_text(encoding="utf-8")
    assert 'record["player_identity"] = identity.record_identity(entry)' in src
    m = re.search(r"^RECORD_SCHEMA_VERSION = (\d+)$", src, re.MULTILINE)
    assert m and int(m.group(1)) >= 10


def test_record_identity_carries_the_declaration_and_the_evidence_side_by_side():
    entry = {
        "player_id": "local-gpt-oss-20b-q8-grim",
        "instrument": "local-gguf-gpt-oss-20b-q8-rlow",
        "openai_temperature": 0.0,
        "openai_extra_body": {"seed": 42, "reasoning_effort": "low", "max_tokens": 4096},
    }
    block = identity.record_identity(entry)
    assert block["instrument"] == "local-gguf-gpt-oss-20b-q8-rlow"
    assert block["inference_config"] == {"temperature": 0.0, "seed": 42, "reasoning_effort": "low"}
    assert block["inference_config_sha256"] is not None


def test_an_undeclared_entry_stamps_none_rather_than_inventing_a_name():
    """Absence is labelled, never inferred from the id or the version -- the
    same rule `provenance.containment` and `usage` already follow."""
    block = identity.record_identity({"player_id": "mystery-grim", "player_version": "x/y-Q8_0"})
    assert block["instrument"] is None
    assert block["inference_config"] is None
    assert block["inference_config_sha256"] is None


def test_audit_records_catches_one_name_covering_two_instruments():
    entries = [{"player_id": "p-grim", "instrument": "m-rlow"}]
    low = identity.record_identity(
        {"player_id": "p-grim", "instrument": "m-rlow",
         "openai_extra_body": {"reasoning_effort": "low"}})
    high = identity.record_identity(
        {"player_id": "p-grim", "instrument": "m-rlow",
         "openai_extra_body": {"reasoning_effort": "high"}})
    clean = [{"player_id": "p-grim", "player_identity": low},
             {"player_id": "p-grim", "player_identity": low}]
    assert identity.audit_records(clean, entries) == []      # two-sided
    dirty = [{"player_id": "p-grim", "player_identity": low},
             {"player_id": "p-grim", "player_identity": high}]
    assert any("two instruments" in p for p in identity.audit_records(dirty, entries))


def test_audit_records_ignores_records_that_predate_the_field():
    """Every published record predates v10. Absence must not be read as a
    disagreement, or the gate would fail on the entire existing corpus."""
    entries = [{"player_id": "p-grim", "instrument": "m-rlow"}]
    assert identity.audit_records([{"player_id": "p-grim"}] * 50, entries) == []


def test_audit_records_catches_a_registry_edited_after_the_run():
    entries = [{"player_id": "p-grim", "instrument": "m-rhigh"}]
    stamped = identity.record_identity(
        {"player_id": "p-grim", "instrument": "m-rlow",
         "openai_extra_body": {"reasoning_effort": "low"}})
    problems = identity.audit_records([{"player_id": "p-grim", "player_identity": stamped}], entries)
    assert any("registry was edited after the run" in p for p in problems)


# ------------------------------------------- the Python record audit is WIRED

def _run_identity_audit(tmp_path, monkeypatch, capsys, records, entries):
    """Drive the REAL `framework audit --identity` over a fake arena.

    Everything the audit does unconditionally beyond --identity (frozen
    fingerprints, formula provenance, host split) is stubbed to a no-op so this
    test measures one thing: whether the identity block reads the records. The
    record audit itself is NOT stubbed -- that is the call site under test.
    """
    import argparse

    import framework.audit as faudit
    from framework import cli as fcli

    root = tmp_path / "arenas"
    (root / "demo-id-v1").mkdir(parents=True, exist_ok=True)  # called twice per test
    monkeypatch.setattr(fcli, "arenas_root", lambda: root)
    monkeypatch.setattr(fcli, "load_registry", lambda _p: entries)
    monkeypatch.setattr(fcli, "registry_path", lambda: tmp_path / "registry.yaml")
    monkeypatch.setattr(faudit, "resolve_task_set_version", lambda _a, _t=None: "v1")
    monkeypatch.setattr(faudit, "load_arena_records", lambda _a, _t="v1": records)
    monkeypatch.setattr(faudit, "frozen_drift", lambda _a: [])
    monkeypatch.setattr(faudit, "formula_provenance_problems", lambda _r, **_k: [])
    monkeypatch.setattr(faudit, "host_split_players",
                        lambda _r, **_k: faudit.HostSplitReport(
                            arena_id="demo-id-v1", ok=True, n_players=1,
                            n_hosts=1, n_unverifiable=0))
    args = argparse.Namespace(arena="demo-id-v1", task_set=None, versions=False,
                              symmetry=False, fresh=False, identity=True,
                              write_fingerprints=False)
    code = fcli._cmd_audit(args)
    return code, capsys.readouterr().out


def test_the_cli_audit_really_reads_the_records(tmp_path, monkeypatch, capsys):
    """`identity.audit_records` used to be called ONLY by tests.

    build-data.mjs ran its Node twin as a build gate, and runner.py carried a
    comment naming the Python function as what audits the declaration -- a
    claim with no call site behind it. Two-sided: a clean set must pass AND a
    set where one instrument name covers two configs must fail the audit.
    """
    entries = [{"player_id": "p-grim", "instrument": "m-rlow",
                "player_type": "ai-model", "player_version": "x", "adapter_class": "A",
                "openai_extra_body": {"reasoning_effort": "low"}}]
    low = identity.record_identity(
        {"instrument": "m-rlow", "openai_extra_body": {"reasoning_effort": "low"}})
    high = identity.record_identity(
        {"instrument": "m-rlow", "openai_extra_body": {"reasoning_effort": "high"}})

    code, out = _run_identity_audit(
        tmp_path, monkeypatch, capsys,
        [{"player_id": "p-grim", "player_identity": low}] * 3, entries)
    assert "3 of 3" in out, out          # it examined the records, and said so
    assert "[PROBLEM]" not in out, out
    assert code == 0, out

    code, out = _run_identity_audit(
        tmp_path, monkeypatch, capsys,
        [{"player_id": "p-grim", "player_identity": low},
         {"player_id": "p-grim", "player_identity": high}], entries)
    assert "two instruments" in out, out
    assert code == 1, f"a real record disagreement did not gate\n{out}"
