"""A tool's version history is a DECLARED LINK, and these are the ways it lies.

WHY THIS FILE EXISTS (2026-09-21). ESCImate shipped 0.7.10 while the registry
declared 0.7.6. The fix was a second entry rather than a relabel, because the 216
published records really were produced by 0.7.6. That left the board with two
entries and nothing connecting them, which is the gap this module closes:
``supersedes: <player_id>`` on the newer entry.

THE MEASUREMENT THAT CHOSE A LINK OVER A GROUPING FIELD. Sorting version strings
gets this exact pair backwards::

    >>> sorted(["0.7.10", "0.7.6"])
    ['0.7.10', '0.7.6']

so any design that derives "latest" from the version string would publish 0.7.6
as the newer ESCImate. A link has an inherent direction and cannot sort wrong.

THE MEASUREMENT THAT CHOSE A DECLARATION OVER A DERIVATION, and this one refuted
a recommendation from the 2026-09-21 consult round rather than confirming one.
The round proposed a rule that would REQUIRE a link between any two non-ai-model
entries sharing an ``adapter_class`` and launch target with different
``player_version``, and recorded it as verified against the live registry. Run
here, that rule fires on SEVEN groups, of which exactly one is a version
succession::

    docling-tables      / docling-vlm-tables         tableformer vs vlm-granite
    docling-text        / docling-vlm-text           tableformer vs vlm-granite
    docpluck-academic   / docpluck-standard          2.4.141-academic vs -standard
    liteparse-default   / liteparse-no-ocr           2.11.1 vs 2.11.1-no-ocr
    regcheck-deepseek / -groq / -openai              three backends, one commit
    spss2rmarkdown / spsstor / stata2r-skranz        three different tools
    escimate            / escimate-0-7-10            <- the only real succession

Six false positives, each of which would have failed every registry load. The
seat marked the claim "Yes -- VERIFIED on today's registry"; it was not, and the
difference is that a configuration sibling bakes its configuration INTO the
version string, so the strings differ without a version differing. Hence
:func:`framework.identity.version_skeleton`, which fires on the one shape a
sibling cannot have, and hence a declaration rather than a derivation.

Both halves are pinned below, because a rule that stops separating the six is a
rule that fails the build on six innocent pairs.
"""
from __future__ import annotations

import pytest

from framework.identity import (
    audit_registry,
    lineage,
    launch_target,
    unlinked_version_successions,
    version_skeleton,
)
from framework.paths import registry_path
from framework.registry import load_registry


def _entry(pid, version, adapter="HttpAdapter", **kw):
    e = {
        "player_id": pid,
        "player_type": "platform",
        "player_version": version,
        "adapter_class": adapter,
        "confidence_strategy": "implicit-1.0",
        "deterministic": True,
        "endpoint": "http://127.0.0.1:9422/api/v1/process-text",
    }
    e.update(kw)
    return e


# --- the string-sort trap this design exists to avoid ------------------------

def test_sorting_version_strings_gets_the_live_pair_backwards():
    """The one-line justification for a link, kept executable.

    If this ever stops being true the argument above needs rewriting, not
    quietly leaving in place as a claim nobody rechecks.
    """
    assert sorted(["0.7.10", "0.7.6"]) == ["0.7.10", "0.7.6"]
    assert sorted(["0.7.10", "0.7.6"])[-1] == "0.7.6"


# --- rule D: a declared link must be capable of being true -------------------

def test_a_link_to_a_player_that_does_not_exist_is_refused():
    problems = audit_registry([_entry("b", "2.0", supersedes="ghost")])
    assert any("not a player_id in this registry" in p for p in problems), problems


def test_a_self_link_is_refused():
    problems = audit_registry([_entry("a", "1.0", supersedes="a")])
    assert any("names itself" in p for p in problems), problems


def test_a_link_across_two_different_adapters_is_refused():
    problems = audit_registry([
        _entry("a", "1.0", adapter="HttpAdapter"),
        _entry("b", "2.0", adapter="RCliAdapter", supersedes="a"),
    ])
    assert any("different adapter_class" in p for p in problems), problems


def test_a_link_between_two_entries_declaring_THE_SAME_version_is_refused():
    """A succession that changes no version is not a succession.

    Note what this does NOT catch, because the distinction cost me a wrong test:
    a configuration sibling declares a DIFFERENT version string
    (``2.4.141-academic`` vs ``2.4.141-standard``) — the configuration is baked
    into the label — so it does not reach this branch. Siblings are kept apart by
    never being DEMANDED to link (see the skeleton tests below), not by rule D.
    Rule D only refuses a link somebody actually wrote between two entries whose
    declared versions are byte-identical.
    """
    problems = audit_registry([
        _entry("a", "2.4.141"),
        _entry("b", "2.4.141", supersedes="a"),
    ])
    assert any("configuration sibling, not a successor" in p for p in problems), problems


def test_a_link_whose_version_LABEL_SHAPE_changed_is_allowed():
    """A real succession may change the shape of the label, so rule D must not
    require the skeletons to match.

    Both of these are on disk in this repo's archives: an unversioned label
    replaced by a versioned one (``poppler-pdftotext`` -> ``poppler-pdftotext-24.08.0``)
    and a wildcard replaced by an exact release (``grobid-0.8.x`` -> ``grobid-0.8.1``).
    A skeleton check here would refuse both.
    """
    assert audit_registry([
        _entry("a", "poppler-pdftotext", instrument="poppler-pdftotext"),
        _entry("b", "poppler-pdftotext-24.08.0", instrument="poppler-pdftotext-24-08-0",
               supersedes="a"),
    ]) == []


def test_two_successors_of_one_entry_are_refused_as_a_fork():
    problems = audit_registry([
        _entry("a", "1.0"),
        _entry("b", "2.0", supersedes="a"),
        _entry("c", "3.0", supersedes="a"),
    ])
    assert any("is superseded by 2 entries" in p for p in problems), problems


def test_a_cycle_is_refused_once_not_once_per_member():
    problems = audit_registry([
        _entry("a", "1.0", supersedes="c"),
        _entry("b", "2.0", supersedes="a"),
        _entry("c", "3.0", supersedes="b"),
    ])
    cycles = [p for p in problems if "forms a cycle" in p]
    assert cycles, problems
    assert len(cycles) == 1, f"a 3-cycle reported {len(cycles)} times: {cycles}"
    # and it names every member, so the reader does not have to walk it by hand
    for member in ("'a'", "'b'", "'c'"):
        assert member in cycles[0], cycles[0]


def test_a_list_instead_of_an_id_is_refused_rather_than_silently_indexed():
    problems = audit_registry([_entry("b", "2.0", supersedes=["a", "x"])])
    assert any("must be a single player_id string" in p for p in problems), problems


def test_a_well_formed_link_produces_no_problems():
    """Negative control. Every assertion above passes on a registry that is
    simply broken everywhere; this one fails if rule D became unconditional."""
    assert audit_registry([
        _entry("a", "0.7.6", instrument="a-0-7-6"),
        _entry("b", "0.7.10", instrument="b-0-7-10", supersedes="a"),
    ]) == []


# --- the skeleton: what may be DEMANDED, and what must never be --------------

def test_the_skeleton_keeps_a_configuration_word_and_drops_the_numbers():
    assert version_skeleton("0.7.6") == version_skeleton("0.7.10") == "#.#.#"
    assert version_skeleton("docpluck-2.4.141-academic") != \
        version_skeleton("docpluck-2.4.141-standard")
    assert version_skeleton("liteparse-2.11.1") != version_skeleton("liteparse-2.11.1-no-ocr")
    assert version_skeleton(None) == ""


def test_a_version_succession_with_no_link_is_demanded_at_publication():
    problems = unlinked_version_successions([
        _entry("a", "0.7.6"),
        _entry("b", "0.7.10"),
    ])
    assert any("declare a `supersedes:` link" in p for p in problems), problems


def test_the_demand_stops_as_soon_as_the_link_is_declared():
    assert unlinked_version_successions([
        _entry("a", "0.7.6"),
        _entry("b", "0.7.10", supersedes="a"),
    ]) == []


@pytest.mark.parametrize("versions", [
    ("docpluck-2.4.141-academic", "docpluck-2.4.141-standard"),
    ("liteparse-2.11.1", "liteparse-2.11.1-no-ocr"),
    ("docling-2.120.3-tableformer-accurate-noocr",
     "docling-2.120.3-vlm-granite-docling-258m"),
    ("regcheck-bdb961d-groq-llama3.1-8b", "regcheck-bdb961d-deepseek-chat"),
])
def test_configuration_siblings_are_never_demanded_to_declare_a_link(versions):
    """The six false positives, as types. A link demanded here fails the build
    for a pair that is not a version history at all."""
    a, b = versions
    assert unlinked_version_successions([_entry("a", a), _entry("b", b)]) == []


def test_an_ai_model_pair_is_out_of_scope():
    """A model id is not a release of a tool we can re-run."""
    assert unlinked_version_successions([
        _entry("a", "claude-haiku-4-5", player_type="ai-model"),
        _entry("b", "claude-haiku-4-5-20251001", player_type="ai-model"),
    ]) == []


def test_a_different_launch_target_is_not_a_succession():
    assert unlinked_version_successions([
        _entry("a", "1.0", endpoint="http://127.0.0.1:9422/x"),
        _entry("b", "2.0", endpoint="http://127.0.0.1:8070/x"),
    ]) == []


def test_launch_target_reports_absence_as_an_answer():
    """A library adapter launches nothing, and that is not the same as an
    unread field -- the key travels with the value so two such entries group by
    their adapter class alone."""
    assert launch_target({"adapter_class": "DocpluckLibraryAdapter"}) == ("", "")
    key, value = launch_target({"endpoint": "http://x/y"})
    assert key == "endpoint" and "http://x/y" in value


# --- walking the chain -------------------------------------------------------

def test_the_chain_comes_back_newest_first_from_either_end():
    entries = [_entry("a", "0.7.6"), _entry("b", "0.7.10", supersedes="a"),
               _entry("c", "0.8.0", supersedes="b")]
    assert lineage(entries, "a") == ["c", "b", "a"]
    assert lineage(entries, "c") == ["c", "b", "a"]
    assert lineage(entries, "b") == ["c", "b", "a"]


def test_a_tool_with_one_version_is_a_chain_of_one_not_an_empty_list():
    """A caller must be able to tell "one version" from "no such player"."""
    assert lineage([_entry("a", "1.0")], "a") == ["a"]
    assert lineage([_entry("a", "1.0")], "nope") == []


def test_walking_a_cycle_terminates():
    """audit_registry refuses a cycle, but `lineage` must not hang on a registry
    that was loaded before the rule existed, or on hand-built input."""
    entries = [_entry("a", "1.0", supersedes="b"), _entry("b", "2.0", supersedes="a")]
    chain = lineage(entries, "a")
    assert 0 < len(chain) <= 2, chain


# --- the live registry -------------------------------------------------------

def test_the_live_registry_declares_every_version_succession_it_has():
    """The publication gate, run against the real file.

    Its non-vacuity control is the next test: this one passes trivially on a
    registry holding no successions at all.
    """
    entries = load_registry(registry_path())
    problems = unlinked_version_successions(entries)
    assert not problems, (
        "players/registry.yaml holds what looks like a tool-version succession "
        "with no declared link:\n  " + "\n  ".join(problems))


def test_the_live_registry_still_contains_the_pair_this_gate_watches():
    """Non-vacuity. If the ESCImate ladder is ever removed or renamed, the test
    above starts passing over nothing and nobody would notice."""
    entries = load_registry(registry_path())
    ids = {e["player_id"] for e in entries}
    assert {"escimate", "escimate-0-7-10"} <= ids, (
        "the ESCImate version pair is gone from the registry; this file's "
        "publication gate now has nothing to check.")
    assert lineage(entries, "escimate") == ["escimate-0-7-19", "escimate-0-7-10", "escimate"], (
        "the declared ladder changed. Newest first is the contract every "
        "consumer of `lineage()` relies on.")


def test_deleting_the_live_link_turns_the_publication_gate_red():
    """Mutation proof on the REAL registry, in memory.

    The green above is only evidence if its red is reachable. Stripping the one
    declared `supersedes:` line from a loaded copy must make the gate fire — done
    on the loaded objects rather than on the file, because three sessions share
    this working tree and a mutated registry on disk would break all of them.
    """
    entries = [dict(e) for e in load_registry(registry_path())]
    stripped = 0
    for e in entries:
        if e.pop("supersedes", None) is not None:
            stripped += 1
    assert stripped >= 1, "no live entry declares supersedes, so nothing was mutated"
    problems = unlinked_version_successions(entries)
    assert problems, (
        "removing every declared link left the publication gate GREEN. It is not "
        "watching the live registry.")
    assert any("escimate" in p for p in problems), problems
