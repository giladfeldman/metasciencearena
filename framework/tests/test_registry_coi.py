"""Conflict-of-interest flag: every maintainer-built tool is flagged, nothing else is.

The site renders a "COI" badge from ``maintainer_coi`` in players/registry.yaml.
These tests stop a new docpluck/escimate/scimeto/... entry from shipping without
it, and stop the flag spreading to a third-party tool.
"""
import copy
from pathlib import Path

import pytest

from framework.registry import (MAINTAINER_TOOL_FAMILIES, RegistryError,
                                coi_mismatches, load_registry)

REPO = Path(__file__).resolve().parents[2]
REGISTRY = REPO / "players" / "registry.yaml"


@pytest.fixture(scope="module")
def entries():
    return load_registry(REGISTRY)


def test_real_registry_has_no_coi_mismatch(entries):
    assert coi_mismatches(entries) == []


def test_real_registry_flags_a_plausible_number_of_players(entries):
    # A zero here would make the mismatch test vacuous (nothing to check).
    flagged = [e["player_id"] for e in entries if e.get("maintainer_coi")]
    assert len(flagged) >= 9, flagged
    assert {"docpluck-academic", "escimate", "scimeto-replication-api", "spss2rmarkdown"} <= set(flagged)


def test_unflagged_maintainer_tool_is_caught(entries):
    broken = copy.deepcopy(entries)
    victim = next(e for e in broken if e["player_id"] == "docpluck-academic")
    del victim["maintainer_coi"]
    problems = coi_mismatches(broken)
    assert len(problems) == 1 and "docpluck-academic" in problems[0]


def test_flag_on_third_party_tool_is_caught(entries):
    broken = copy.deepcopy(entries)
    victim = next(e for e in broken if e["player_id"] == "statcheck")
    victim["maintainer_coi"] = True
    problems = coi_mismatches(broken)
    assert len(problems) == 1 and "statcheck" in problems[0]


def test_family_match_is_on_instrument_boundary():
    # "scimetoolkit-1-0" must not be read as a scimeto instrument.
    entry = {"player_id": "x", "instrument": "scimetoolkit-1-0"}
    assert coi_mismatches([entry]) == []
    assert "scimeto" in MAINTAINER_TOOL_FAMILIES


def test_non_bool_flag_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "- {player_id: x, player_type: tool, player_version: '1', adapter_class: A, "
        "confidence_strategy: native, deterministic: true, maintainer_coi: 'yes'}\n",
        encoding="utf-8",
    )
    with pytest.raises(RegistryError):
        load_registry(bad)
