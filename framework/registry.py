"""Load + validate player registry."""
from __future__ import annotations

from pathlib import Path

import yaml

from framework import identity

REQUIRED_FIELDS = {"player_id", "player_type", "player_version", "adapter_class", "confidence_strategy", "deterministic"}
ALLOWED_TYPES = {"platform", "tool", "human-baseline", "ai-model"}
ALLOWED_CONFIDENCE = {"native", "implicit-1.0", "derived"}

# CONFLICT OF INTEREST (2026-09-27). Tools built and maintained by the person
# who runs this arena. Every registry entry whose declared ``instrument`` belongs
# to one of these families must carry ``maintainer_coi: true`` and no other entry
# may; the site renders a "COI" badge from that flag and links it to
# /about#conflict-of-interest. The flag is what the site reads; this list is only
# the cross-check that stops a new docpluck/escimate/... entry shipping unflagged.
# `effectcheck` is the R engine inside every escimate-* pipeline player of
# stats-from-pdf-v1; those entries declare instrument `effectcheck-<ver>-on-<extractor>`.
# Families with no registry entry yet (citelink, referencecheck,
# torturedcheck, repliscan, articlefinder) are listed on purpose: they are the
# maintainer's tools most likely to be entered next (see
# docs/design/2026-09-arena-expansion/), and must arrive flagged.
MAINTAINER_TOOL_FAMILIES = ("docpluck", "escimate", "effectcheck", "scimeto", "spss2rmarkdown",
                            "citelink", "referencecheck",
                            "torturedcheck", "repliscan", "articlefinder")


class RegistryError(Exception):
    pass


#: Quarantine-by-default for new players (docs/recovery-regime.md, Phase 3),
#: implemented 2026-10-07. A quarantined player is PUBLISHED and visible -- the
#: board shows its row with a "Provisional" badge -- but is declared not yet
#: audited. `quarantine:` is a mapping {since: "YYYY-MM-DD", reason: str}; it is
#: lifted by REPLACING it with `quarantine_released: {on: "YYYY-MM-DD", by: str}`
#: once a blinded audit has checked the player's records, so the release is on
#: the record rather than a silently deleted line.
#: Gate: framework/tests/test_new_players_are_quarantined.py.
QUARANTINE_ADOPTED_ON = "2026-10-07"


def quarantine_problem(entry: dict) -> str | None:
    import re

    q, rel = entry.get("quarantine"), entry.get("quarantine_released")
    if q is not None and rel is not None:
        return "declares both quarantine and quarantine_released; a release replaces the quarantine"
    if q is not None:
        if not isinstance(q, dict) or not isinstance(q.get("reason"), str) or not q["reason"].strip():
            return "quarantine must be a mapping with a non-empty `reason`"
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(q.get("since", ""))):
            return "quarantine.since must be a YYYY-MM-DD date"
    if rel is not None:
        if not isinstance(rel, dict) or not isinstance(rel.get("by"), str) or not rel["by"].strip():
            return "quarantine_released must be a mapping with a non-empty `by`"
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(rel.get("on", ""))):
            return "quarantine_released.on must be a YYYY-MM-DD date"
    return None


def load_registry(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or []
    if not isinstance(data, list):
        raise RegistryError(f"{path}: registry must be a YAML list")

    seen_ids: set[str] = set()
    for entry in data:
        if not isinstance(entry, dict):
            raise RegistryError(f"{path}: each entry must be a mapping; got {type(entry).__name__}")
        missing = REQUIRED_FIELDS - entry.keys()
        if missing:
            raise RegistryError(f"{path}: entry missing fields {sorted(missing)}: {entry}")
        if entry["player_type"] not in ALLOWED_TYPES:
            raise RegistryError(f"{path}: player_type must be one of {sorted(ALLOWED_TYPES)}, got {entry['player_type']}")
        if entry["confidence_strategy"] not in ALLOWED_CONFIDENCE:
            raise RegistryError(f"{path}: confidence_strategy must be one of {sorted(ALLOWED_CONFIDENCE)}, got {entry['confidence_strategy']}")
        if entry["player_id"] in seen_ids:
            raise RegistryError(f"{path}: duplicate player_id {entry['player_id']}")
        seen_ids.add(entry["player_id"])
        if "maintainer_coi" in entry and not isinstance(entry["maintainer_coi"], bool):
            raise RegistryError(f"{path}: {entry['player_id']}: maintainer_coi must be true/false, got {entry['maintainer_coi']!r}")
        if "coi_note" in entry and not isinstance(entry["coi_note"], str):
            raise RegistryError(f"{path}: {entry['player_id']}: coi_note must be a string")
        problem = quarantine_problem(entry)
        if problem:
            raise RegistryError(f"{path}: {entry['player_id']}: {problem}")

    # DECLARED INSTRUMENT IDENTITY (2026-09-17). Deliberately NOT in
    # REQUIRED_FIELDS: a second machine's registry is edited independently and
    # must keep loading while its lines are being written. What is enforced here
    # is that a declaration, once present, is CONSISTENT -- see
    # framework.identity.audit_registry for the three rules and the failure each
    # one catches. Requiring the field for anything that PUBLISHES is a separate
    # gate (framework audit), because that is where the cost of a missing
    # declaration is actually paid.
    problems = identity.audit_registry(data)
    if problems:
        raise RegistryError(
            f"{path}: declared instrument identity is inconsistent:\n  "
            + "\n  ".join(problems))

    return data


def undeclared_instruments(entries: list[dict]) -> list[str]:
    """player_ids with no ``instrument:`` line, sorted.

    Its own function rather than a flag inside :func:`load_registry` because the
    two questions have different answers: loading a registry mid-edit is fine,
    PUBLISHING a player whose instrument nobody named is not -- that player's
    /players row would have to fall back to the normalised ``player_version``,
    which is the exact key this design replaced.
    """
    return sorted(e["player_id"] for e in entries if not e.get("instrument"))


def coi_mismatches(entries: list[dict]) -> list[str]:
    """Entries whose ``maintainer_coi`` flag disagrees with MAINTAINER_TOOL_FAMILIES.

    Matched on the DECLARED ``instrument`` (``<family>-<version>``), never on the
    player_id, which is free text. An entry with no instrument cannot be checked
    and is reported, so a missing declaration cannot pass as "not ours".
    """
    problems = []
    for e in entries:
        inst = e.get("instrument")
        flagged = e.get("maintainer_coi") is True
        if not inst:
            if flagged:
                problems.append(f"{e['player_id']}: maintainer_coi set but no instrument declared")
            continue
        ours = any(inst == f or inst.startswith(f + "-") for f in MAINTAINER_TOOL_FAMILIES)
        if ours and not flagged:
            problems.append(f"{e['player_id']}: instrument {inst} is a maintainer tool but maintainer_coi is not true")
        elif flagged and not ours:
            problems.append(f"{e['player_id']}: maintainer_coi is true but instrument {inst} is not a maintainer tool")
    return problems
