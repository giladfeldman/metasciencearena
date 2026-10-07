"""WHICH INSTRUMENT PRODUCED A NUMBER, as opposed to which label was on it.

WHY THIS MODULE EXISTS (2026-09-17)
-----------------------------------
``player_version`` labels WEIGHTS. It does not record INFERENCE CONFIGURATION,
and inference configuration is the largest single effect measured in this
corpus. On the second machine 35 distinct builds collapse to 18 ``player_version``
strings, because the same weights file run with thinking on and thinking off
declares one identical string::

    local-gguf/Qwen3.6-35B-A3B-UD-Q3_K_XL  -> q3, q3think, q3thinkhc
    local-gguf/gpt-oss-20b-Q8_0            -> q8, q8-rmed, q8-rhigh

The q3 / q3think pair differs by **+0.24 to +0.28 mean score** -- larger than any
model-choice effect here. A hub keyed on ``player_version`` renders them as ONE
card, and a deliberate reasoning-effort dose-response ladder is flattened into a
single row.

THE DESIGN: DECLARE, THEN AUDIT
-------------------------------
One hand-written line per registry entry names the real instrument::

    - player_id: local-gpt-oss-20b-q8-grim
      player_version: "local-gguf/gpt-oss-20b-Q8_0"
      instrument: local-gpt-oss-20b-q8-rlow      # <- declared, not parsed

and the settings actually sent are stamped onto NEW run records
(``player_identity``, schema v10), so the declaration can be checked against
reality instead of believed.

WHY NOT A PARSE, A NORMALISER, OR A HASH
----------------------------------------
``leaderboard-app/lib/model-family.ts`` already made the argument against parsing
``player_id``: "Any parse of the id is therefore wrong for a population we can
enumerate." That argument transfers wholesale to ``player_version``, and this
repo has the receipts for both halves:

* **A normaliser cannot separate what the string does not distinguish.** q3 and
  q3think declare the same bytes. No normaliser recovers a difference that was
  never written down.
* **A normaliser can MERGE what should stay apart.** ``familySlug`` squeezes
  every non-alphanumeric run to "-", so ".../GLM-4.7-Flash/UD-Q4_K_XL" and
  ".../GLM-4.7-Flash-UD-Q4_K_XL" produce ONE url,
  "local-gguf-glm-4-7-flash-ud-q4-k-xl". Those two players are the non-thinking
  and thinking arms of the same weights. Two of three consult seats checked only
  ``normaliseVersion``, found the pair distinct, and gave a confident all-clear;
  the third checked the second function in the same file and found the collision
  (2026-09-16 round, ``docs/consults/2026-09-16-player-identity/``).
* **Repairing the separator does not help.** The second machine repaired it and
  the collision SURVIVED, because the two strings then became identical -- same
  weights, and ``player_version`` labels weights. A separator accident surfaced a
  collision that is real, and remains after the accident is repaired.
* **The weights digest is too coarse.** Same sha256, thinking on vs off, +0.28
  mean. ``gguf_sha256`` is excellent evidence of WHICH FILE was loaded and is not
  the instrument. It also does not exist for hosted players.

Declared instrument strings are constrained to be slug-safe
(:data:`INSTRUMENT_RE`), so ``familySlug(instrument) == instrument`` and the url
collision above cannot be constructed at all. That is the point of declaring: the
defect is removed at its cause rather than patched in the normaliser.

WHAT IS *NOT* IDENTITY
----------------------
The HOST is provenance, not identity -- all three consult seats agreed, and the
schema already carries ``provenance.host_fingerprint`` for it. Two machines
running the same instrument are a REPLICATION, not two models. See
:func:`framework.host.host_identity`, and the note in
``players/adapters/local_gguf.py`` about what the host belongs to
(``resolved_tool_version``, which is evidence of what ran) and what it does not
(the instrument a reader is shown).
"""
from __future__ import annotations

import hashlib
import json
import re

#: A declared instrument is slug-safe BY CONSTRUCTION. Lowercase alphanumerics
#: joined by single hyphens -- the exact output shape of `familySlug`, so that
#: function is the identity on an instrument string and two distinct instruments
#: can never collapse to one /models/<slug> url.
INSTRUMENT_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

#: Longest instrument we accept. Not a technical limit -- a bound that keeps the
#: string readable in a table cell and in a url.
INSTRUMENT_MAX_LEN = 80

#: Request keys that are a BUDGET rather than a configuration of the model.
#: `max_tokens` is score-affecting (a ceiling below the model's own published a
#: perfect 1.000 as 0.0 -- see run_record.schema.json provenance.request_max_tokens)
#: but it is recorded in its own field on every record, and folding it in here
#: would split an instrument every time a ceiling is corrected. Excluded on the
#: same reasoning `players/adapters/local_gguf.py` already excludes it from
#: `response_meta.sampling`.
_BUDGET_KEYS = frozenset({"max_tokens"})


def inference_config(entry: dict) -> dict | None:
    """The score-affecting request knobs this registry entry sends, canonicalised.

    ``None`` is a FIRST-CLASS ANSWER and means "this player exposes no inference
    settings to us" -- a deterministic R tool, a library parser, a CLI whose
    reasoning mode is baked into its own argv. It never means "we did not look".
    A reader must not treat it as an empty configuration: an R script has no
    temperature, and asserting one would be a claim nobody measured.

    An empty dict is impossible by construction: if any knob is present the dict
    is non-empty, and if none is, the result is ``None``.
    """
    body = entry.get("openai_extra_body") or {}
    temp = entry.get("openai_temperature")
    if temp is None and not body:
        return None
    cfg: dict = {}
    if temp is not None:
        cfg["temperature"] = temp
    for k, v in body.items():
        if k in _BUDGET_KEYS:
            continue
        cfg[k] = v
    return cfg or None


def canonical_json(cfg: dict | None) -> str:
    """Stable text for a config, so two equal configs hash equal.

    ``sort_keys`` recurses, so a nested block written in a different key order
    (``chat_template_kwargs``) still canonicalises to one string.
    """
    return json.dumps(cfg, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def inference_config_sha256(cfg: dict | None) -> str | None:
    """Digest of a config, or ``None`` when there is no config to digest.

    ``None`` in, ``None`` out -- deliberately NOT the hash of the string "null".
    Hashing the absence would make "sends no settings" indistinguishable from a
    settings block that happens to be recorded, and the whole point of the absent
    case is that it is a different answer.
    """
    if cfg is None:
        return None
    return hashlib.sha256(canonical_json(cfg).encode("utf-8")).hexdigest()


def record_identity(entry: dict) -> dict:
    """The ``player_identity`` block to stamp on a run record (schema v10).

    Copies the DECLARED instrument beside the config actually sent, so a later
    reader can check one against the other without holding the registry that
    produced the run. ``instrument`` is ``None`` when the entry declares none --
    labelled, never inferred from the id or the version.
    """
    cfg = inference_config(entry)
    return {
        "instrument": entry.get("instrument"),
        "inference_config": cfg,
        "inference_config_sha256": inference_config_sha256(cfg),
    }


def audit_registry(entries: list[dict]) -> list[str]:
    """Check declared instruments against the configs they claim to name.

    Returns a list of human-readable violations; empty means clean. Three rules,
    each of which has a failure it exists to catch:

    **A. One instrument, one inference config.** The declaration is only worth
    anything if it cannot hide a config difference. This is the rule that would
    have caught q3 and q3think being filed under one name.

    **B. One (version, config, adapter), one instrument.** The reverse: two
    entries that are the same instrument in every observable way must not be
    declared as two, or one card silently becomes two rows. ``adapter_class`` is
    in the key because a PDF-sections harness around Claude and a plain CLI
    harness around Claude are the same model reached by different arena plumbing,
    and the operator -- not a rule -- decides whether that is one instrument.
    Rule B therefore never forces those apart or together; it only refuses a
    split among entries that are identical down to the adapter.

    **C. Slug-safe by construction.** See :data:`INSTRUMENT_RE`.

    **D. A declared ``supersedes`` link must be capable of being true.** The
    version-history axis (2026-09-21) is expressed as one optional line on the
    NEWER entry naming the id it replaces. It is a LINK, not a group, and not a
    parse, for a reason measured on this registry: sorting version strings puts
    ``"0.7.10"`` BEFORE ``"0.7.6"``, so anything deriving "latest" from the
    string gets the live ESCImate case backwards. A link cannot sort wrong.
    Rule D refuses a link that could never describe a succession -- a dangling
    id, a self-reference, a cycle, a different adapter, or two entries declaring
    the same ``player_version`` (which is a configuration sibling, not a
    successor). It never INVENTS a link; see
    :func:`unlinked_version_successions` for where absence is chased, and why
    that is not here.

    Note what is deliberately NOT a rule: one instrument spanning several
    declared ``player_version`` strings is ALLOWED. Labels drift -- that is the
    premise of this whole module -- and /players already surfaces every declared
    string it merged ("Declared as ..."), so the fact stays visible instead of
    being enforced away.
    """
    problems: list[str] = []
    problems.extend(_supersedes_problems(entries))
    by_instrument: dict[str, list[dict]] = {}
    by_shape: dict[tuple, list[dict]] = {}

    for e in entries:
        inst = e.get("instrument")
        if inst is None:
            continue
        if not isinstance(inst, str) or not INSTRUMENT_RE.match(inst):
            problems.append(
                f"{e.get('player_id')}: instrument {inst!r} is not slug-safe. "
                "Required shape: lowercase alphanumerics joined by single hyphens.")
            continue
        if len(inst) > INSTRUMENT_MAX_LEN:
            problems.append(
                f"{e.get('player_id')}: instrument {inst!r} is {len(inst)} "
                f"characters, over the {INSTRUMENT_MAX_LEN}-character bound.")
            continue
        by_instrument.setdefault(inst, []).append(e)
        shape = (e.get("player_version"), canonical_json(inference_config(e)),
                 e.get("adapter_class"))
        by_shape.setdefault(shape, []).append(e)

    for inst, group in sorted(by_instrument.items()):
        configs = {canonical_json(inference_config(e)) for e in group}
        if len(configs) > 1:
            ids = ", ".join(sorted(str(e.get("player_id")) for e in group))
            problems.append(
                f"instrument {inst!r} is declared on entries with {len(configs)} "
                f"DIFFERENT inference configs: {sorted(configs)}. Entries: {ids}. "
                "Same instrument means same settings -- give the arms distinct "
                "instrument names (e.g. ...-rlow / ...-rhigh).")

    for shape, group in sorted(by_shape.items(), key=lambda kv: str(kv[0])):
        names = {e["instrument"] for e in group}
        if len(names) > 1:
            ids = ", ".join(sorted(str(e.get("player_id")) for e in group))
            problems.append(
                "entries identical in player_version, inference config AND "
                f"adapter_class are declared as {len(names)} different instruments "
                f"{sorted(names)}: {ids}. One of these names is a split that would "
                "publish one instrument as two rows.")

    return problems


def audit_records(records: list[dict], entries: list[dict]) -> list[str]:
    """Check stamped records against the registry that claims to describe them.

    Only records carrying ``player_identity`` are examined -- its absence means
    the record predates schema v10 and is never inferred. Records are the
    evidence the registry is a claim about, so a disagreement is always reported
    rather than resolved in the registry's favour.
    """
    problems: list[str] = []
    declared = {e["player_id"]: e for e in entries if "player_id" in e}
    seen: dict[str | None, set[str | None]] = {}

    for r in records:
        ident = r.get("player_identity")
        if not ident:
            continue
        pid = r.get("player_id")
        inst = ident.get("instrument")
        seen.setdefault(inst, set()).add(ident.get("inference_config_sha256"))
        entry = declared.get(pid)
        if entry is None:
            continue
        want = entry.get("instrument")
        if want is not None and inst is not None and want != inst:
            problems.append(
                f"{pid}: a record stamps instrument {inst!r} but the registry now "
                f"declares {want!r}. The registry was edited after the run; decide "
                "which is right rather than letting the page and the record disagree.")

    for inst, hashes in sorted(seen.items(), key=lambda kv: str(kv[0])):
        if inst is not None and len(hashes) > 1:
            problems.append(
                f"instrument {inst!r} appears on records with {len(hashes)} "
                f"different inference-config digests {sorted(map(str, hashes))}. "
                "One name is covering two instruments in PUBLISHED data.")

    return problems


# ---------------------------------------------------------------------------
# VERSION LINEAGE (2026-09-21)
#
# A tool ships a new release; the arena re-runs it and the two results must be
# readable as one product's history rather than as two unrelated players. The
# existing keys cannot express that: `player_id` is unique per entry,
# `player_version` is a label stamped on records, and `instrument` is
# deliberately version-bearing for tools (`escimate-0-7-6`, `statcheck-1-5-0`,
# `grobid-0-8-1`) precisely so one row cannot silently absorb a second version.
#
# WHY A DECLARED LINK AND NOT A GROUPING KEY OR A DERIVATION. All three were put
# to a consult on 2026-09-21 (docs/consults/2026-09-21-tool-version-history/) and
# then measured here:
#
#   * A grouping field (`lineage: escimate` on both entries) gives membership but
#     no ORDER, and order would still come from the version string --
#     `sorted(["0.7.10", "0.7.6"])` is `['0.7.10', '0.7.6']`, so "latest" would
#     be 0.7.6. Record timestamps do not save it either: an old version re-run
#     later becomes the newest record.
#   * A derivation from the registry's own fields was REFUTED by measurement.
#     Grouping the 42 non-ai-model entries by (adapter_class, launch target)
#     yields seven groups holding more than one declared version, of which
#     exactly ONE is a version succession (the escimate pair). The other six are
#     configuration siblings whose version strings differ because the
#     configuration is baked into them: docpluck academic/standard, docling
#     tableformer/vlm, liteparse default/no-ocr, three regcheck backends and
#     three entirely different xlat tools. A derivation cannot tell "same tool,
#     new version" from "same tool, other configuration", which is
#     `leaderboard-app/lib/model-family.ts`'s argument re-measured on tools.
#
# So the link is written by hand, and everything here only checks it.
# ---------------------------------------------------------------------------

#: Registry keys that identify WHAT an entry launches, in the order a lookup
#: should prefer them. Two entries with the same adapter class and the same
#: launch target are pointed at the same thing.
_LAUNCH_KEYS = ("endpoint", "start_command", "r_script", "cli_command",
                "api_url_env", "catalog_path")


def launch_target(entry: dict) -> tuple[str, str]:
    """``(key, canonical value)`` naming what this entry launches.

    ``("", "")`` when the entry declares none -- a library adapter whose target
    is the adapter class itself. That is a first-class answer, and it is why the
    key travels with the value: two entries that both declare nothing are then
    grouped by their adapter class alone, which is correct for a library.
    """
    for key in _LAUNCH_KEYS:
        if entry.get(key) is not None:
            return (key, canonical_json(entry[key]))
    return ("", "")


def version_skeleton(version: str | None) -> str:
    """``player_version`` with every run of digits replaced by ``#``.

    USED ONLY TO PROMPT A DECLARATION, NEVER TO DECIDE IDENTITY. The distinction
    is the whole reason this module exists: a parse of a version string cannot
    establish what a player IS. But it can notice that two entries differ ONLY in
    numbers, which is the one shape that cannot be a configuration sibling -- a
    sibling always carries a configuration WORD (``-academic``/``-standard``,
    ``-no-ocr``, a provider name), so its skeleton differs. Measured 2026-09-21
    across all 42 non-ai-model entries: this separates every one of the six known
    sibling groups and leaves exactly the one true succession.
    """
    return re.sub(r"\d+", "#", version or "")


def _supersedes_problems(entries: list[dict]) -> list[str]:
    """Rule D. See :func:`audit_registry`."""
    problems: list[str] = []
    by_id = {e["player_id"]: e for e in entries
             if isinstance(e, dict) and "player_id" in e}

    superseded_by: dict[str, list[str]] = {}
    for e in entries:
        if not isinstance(e, dict):
            continue
        target = e.get("supersedes")
        if target is None:
            continue
        pid = e.get("player_id")
        if not isinstance(target, str):
            problems.append(
                f"{pid}: supersedes must be a single player_id string, got "
                f"{type(target).__name__}. A fork is not a succession: if two "
                "entries really replace one, neither is 'the latest' and the "
                "history is a tree nobody can render as a ladder.")
            continue
        if target == pid:
            problems.append(f"{pid}: supersedes names itself.")
            continue
        older = by_id.get(target)
        if older is None:
            problems.append(
                f"{pid}: supersedes {target!r}, which is not a player_id in this "
                "registry. A link to a retired entry is a link to nothing -- keep "
                "the predecessor's entry as a tombstone or drop the line.")
            continue
        if e.get("adapter_class") != older.get("adapter_class"):
            problems.append(
                f"{pid}: supersedes {target!r}, but they declare different "
                f"adapter_class ({e.get('adapter_class')!r} vs "
                f"{older.get('adapter_class')!r}). Two versions of one tool are "
                "reached the same way; a different adapter is a different "
                "instrument, however similar the name.")
        if e.get("player_version") == older.get("player_version"):
            problems.append(
                f"{pid}: supersedes {target!r}, but both declare player_version "
                f"{e.get('player_version')!r}. Same version is a configuration "
                "sibling, not a successor -- docpluck academic/standard and "
                "liteparse default/no-ocr are the shape this refuses.")
        superseded_by.setdefault(target, []).append(str(pid))

    for target, successors in sorted(superseded_by.items()):
        if len(successors) > 1:
            problems.append(
                f"{target!r} is superseded by {len(successors)} entries "
                f"{sorted(successors)}. A ladder has one head; two successors is "
                "a fork, and picking one to show is selection after observing "
                "the result.")

    # Cycles. A ladder that loops has no head, and a consumer walking it either
    # hangs or reports whichever arbitrary node it happened to start from.
    #
    # Reported ONCE PER CYCLE, keyed on the cycle's member set. Keying on the
    # node where the walk re-entered the loop reports an N-cycle N times, once
    # per starting point -- which reads as N separate defects and buries the
    # single edit that fixes them all.
    reported: set[frozenset] = set()
    for start in sorted(by_id):
        seen: set[str] = set()
        node: str | None = start
        while node is not None:
            if node in seen:
                members, walk = [], node
                while True:
                    members.append(walk)
                    nxt = by_id.get(walk, {}).get("supersedes")
                    walk = nxt if isinstance(nxt, str) and nxt in by_id else None
                    if walk is None or walk == node:
                        break
                key = frozenset(members)
                if key not in reported:
                    reported.add(key)
                    problems.append(
                        f"supersedes forms a cycle through {sorted(members)}. "
                        "Break it: a version history with no oldest entry cannot "
                        "be ordered.")
                break
            seen.add(node)
            nxt = by_id.get(node, {}).get("supersedes")
            node = nxt if isinstance(nxt, str) and nxt in by_id else None

    return problems


def unlinked_version_successions(entries: list[dict]) -> list[str]:
    """Entry groups that LOOK like a version succession and declare no link.

    Its own function rather than a rule inside :func:`audit_registry`, for the
    same reason :func:`undeclared_instruments` is one: a registry mid-edit must
    keep LOADING, and a second machine adds entries for weeks before they
    publish. Absence of a link is chased where the cost is paid -- at
    publication -- not at import time.

    The test is deliberately narrow, because a false positive here fails a build
    for a pair that is not a succession at all. It requires ALL of: the same
    ``adapter_class``, the same :func:`launch_target`, the same
    :func:`version_skeleton`, and a different ``player_version``. Measured
    2026-09-21 on the live registry, that fires on one group (the ESCImate pair)
    and on none of the six configuration-sibling groups a coarser rule caught.

    ``ai-model`` entries are out of scope: a model id is not a release of a tool
    we can re-run, and /players already groups those by declared instrument.
    """
    groups: dict[tuple, list[dict]] = {}
    for e in entries:
        if not isinstance(e, dict) or e.get("player_type") == "ai-model":
            continue
        key = (e.get("adapter_class"), launch_target(e),
               version_skeleton(e.get("player_version")))
        groups.setdefault(key, []).append(e)

    problems: list[str] = []
    for key, group in sorted(groups.items(), key=lambda kv: str(kv[0])):
        if len(group) < 2:
            continue
        if len({e.get("player_version") for e in group}) < 2:
            continue
        ids = sorted(str(e.get("player_id")) for e in group)
        linked = {str(e.get("player_id")) for e in group
                  if isinstance(e.get("supersedes"), str)}
        # Exactly one entry per chain is the oldest and correctly carries no link.
        if len(linked) >= len(ids) - 1:
            continue
        problems.append(
            f"{ids} are the same {key[0]} pointed at the same target and differ "
            f"only in the numbers of their player_version "
            f"({sorted(str(e.get('player_version')) for e in group)}), but only "
            f"{len(linked)} of {len(ids)} declare a `supersedes:` link. Add one "
            "line to the newer entry naming the id it replaces, so a page can "
            "order them without parsing a version string. If they are NOT "
            "versions of each other, give them version labels that say so -- a "
            "configuration belongs in the label, the way "
            "docpluck-2.4.141-academic carries it.")
    return problems


def lineage(entries: list[dict], player_id: str) -> list[str]:
    """The whole chain containing ``player_id``, NEWEST FIRST.

    Walks declared links only. A player with no link in either direction is a
    one-element chain, which is the right answer for the ~40 tools that have only
    ever run one version here -- never an empty list, which a caller would then
    have to tell apart from "unknown player". An unknown player is ``[]``.
    """
    by_id = {e["player_id"]: e for e in entries
             if isinstance(e, dict) and "player_id" in e}
    if player_id not in by_id:
        return []

    newer: dict[str, str] = {}
    for pid, e in by_id.items():
        target = e.get("supersedes")
        if isinstance(target, str):
            newer[target] = pid

    head, seen = player_id, {player_id}
    while head in newer and newer[head] not in seen:
        head = newer[head]
        seen.add(head)

    chain: list[str] = []
    node: str | None = head
    while node is not None and len(chain) <= len(by_id):
        chain.append(node)
        nxt = by_id.get(node, {}).get("supersedes")
        node = nxt if isinstance(nxt, str) and nxt in by_id and nxt not in chain else None
    return chain
