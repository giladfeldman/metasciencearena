"""Generator for reference-integrity-v1 (metacheck-style arena).

Builds a paper's reference list with KNOWN integrity defects injected per
reference, mirroring metacheck's ref_* integrity modules. All-procedural and
deterministic from (task_set_version, seed).

The arena KNOWS the ground truth because it injected every defect itself: it made
a reference "retracted", altered metadata away from a stored canonical value,
dropped an in-text marker's reference, corrupted a DOI, swapped a reputable venue
for a predatory one, or mangled a title into a paper-mill tortured phrase. No live
retraction/metadata/Beall database is required.

Dual-benchmark (revealed/private): both splits run the IDENTICAL tier matrix and
assign the injected issue KIND deterministically (index-driven cycling through
ALL issue kinds, seed-independent), so every split covers the full array of
injected issues equally — this is what makes framework/parity.py pass. Only the
concrete reference CONTENT (which catalog entry, which altered value) is
seed-driven, so revealed and private content still differ.

Realism / clean controls: every new mistake kind ships with a matched CLEAN
look-alike (a valid-but-unusual DOI, a legitimate-but-obscure journal, real
domain jargon) rendered in the controls (T2) and subtle (T4) tiers. A good player
must flag the defect WITHOUT false-alarming the honest look-alike.

Gold is regenerated from the seed and served from the in-process cache (no
external registry): the secret is the private seed, not a stored answer key. The
revealed seed is committed in arena.yaml#benchmark_splits.
"""
from __future__ import annotations

import hashlib
import random
from pathlib import Path

import yaml

ARENA_DIR = Path(__file__).resolve().parent
CATALOGS_DIR = ARENA_DIR / "catalogs"

_GROUND_TRUTH_CACHE: dict[str, dict] = {}

# The closed set of injected integrity issue kinds, in canonical cycle order.
# Index-driven cycling through THIS list (not rng.choice) is what guarantees both
# splits cover every issue kind => parity passes.
ISSUE_KINDS = [
    "retracted",
    "metadata_mismatch",
    "dangling_uncited",
    "dangling_missing",
    "replication_uncited",
    "miscitation",
    "invalid_doi",
    "predatory_source",
    "tortured_phrase",
]

# Issue kinds that need a catalog entry carrying a known_replication block.
_NEEDS_REPLICATION = {"replication_uncited"}
# Issue kinds that need a catalog entry carrying miscite_* fields.
_NEEDS_MISCITE = {"miscitation"}
# Issue kinds that need a catalog entry carrying a predatory_venue twin.
_NEEDS_PREDATORY = {"predatory_source"}
# Issue kinds that need a catalog entry carrying a tortured_title twin.
_NEEDS_TORTURED = {"tortured_phrase"}

# Clean-control look-alike pools (keys of catalogs/references.yaml#clean_controls)
# mapped to the mistake kind whose honest twin they are. A "control:<pool>" plan
# slot renders one of these as a CLEAN reference that nonetheless LOOKS suspicious
# to a naive checker — flagging it is a false alarm.
_CONTROL_POOLS = {
    "legit_unusual_doi": "invalid_doi",
    "obscure_venue": "predatory_source",
    "legit_jargon_title": "tortured_phrase",
}


def _load_catalog() -> list[dict]:
    with (CATALOGS_DIR / "references.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_clean_controls() -> dict[str, list[dict]]:
    """Load the clean-control look-alike pools (the honest twins of new kinds)."""
    with (CATALOGS_DIR / "clean_controls.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _seed_int(*parts) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def _pick_entry(catalog: list[dict], rng: random.Random, kind: str, used: set) -> dict:
    """Pick a catalog entry suitable for `kind`, avoiding ids already used here."""
    pool = catalog
    if kind in _NEEDS_REPLICATION:
        pool = [c for c in catalog if c.get("known_replication")]
    elif kind in _NEEDS_MISCITE:
        pool = [c for c in catalog if c.get("miscite_year") and c.get("miscite_title")]
    elif kind in _NEEDS_PREDATORY:
        pool = [c for c in catalog if c.get("predatory_venue")]
    elif kind in _NEEDS_TORTURED:
        pool = [c for c in catalog if c.get("tortured_title")]
    candidates = [c for c in pool if c["id"] not in used] or pool
    return rng.choice(candidates)


def _canonical_record(entry: dict, ref_id: str) -> dict:
    return {
        "reference_id": ref_id,
        "authors": list(entry["authors"]),
        "year": entry["year"],
        "title": entry["title"],
        "doi": entry["doi"],
        "venue": entry.get("venue", "Unknown Journal"),
    }


def _corrupt_doi(doi: str, rng: random.Random) -> str:
    """Return a structurally INVALID variant of a DOI (the arena injected it).

    Deterministic given `rng`. Produces one of a few realistic malformations a
    DOI validator would reject: a letter-O for a zero, a dropped "10." prefix, or
    a truncated suffix. The canonical DOI is stored in gold for verification.
    """
    style = rng.choice(["letter_o", "drop_prefix", "truncate"])
    if style == "letter_o" and "0" in doi:
        # Replace the first zero with a capital letter O (a classic OCR/typo
        # corruption that breaks the DOI but looks plausible at a glance).
        idx = doi.index("0")
        return doi[:idx] + "O" + doi[idx + 1:]
    if style == "drop_prefix" and doi.startswith("10."):
        return doi[3:]  # no "10." registrant prefix => not a resolvable DOI
    # truncate: lop the suffix so the DOI can no longer resolve.
    cut = max(5, len(doi) // 2)
    return doi[:cut]


def _render_reference(entry: dict, ref_id: str, kind: str | None, rng: random.Random):
    """Render one reference dict + its gold record + any extra in_text markers.

    Returns (reference_dict_or_None, extra_marker_ids, gold_record).

    For most kinds the reference is listed (and cited). Two kinds are special:
      - dangling_uncited : reference listed but cited_in_text=False (no marker).
      - dangling_missing : NO reference is listed, but an in-text marker id with no
                           matching reference is emitted; gold is keyed by that
                           marker id. (reference_dict is None.)
    """
    ref = _canonical_record(entry, ref_id)
    ref["cited_in_text"] = True
    gold = {"reference_id": ref_id, "issue_kind": None, "flagged": False}

    if kind is None or kind == "clean":
        return ref, [], gold

    if kind == "retracted":
        # The arena marks the DOI retracted (it knows; it injected the flag).
        ref["doi"] = ref["doi"] + "  [RETRACTED]"
        gold = {"reference_id": ref_id, "issue_kind": "retracted", "flagged": True}
        return ref, [], gold

    if kind == "metadata_mismatch":
        # Alter ONE bibliographic field away from canonical. The canonical value is
        # stored in gold so a scorer/oracle could verify the corrected value.
        field = rng.choice(["authors", "year", "title"])
        if field == "authors":
            ref["authors"] = [a for a in ref["authors"][::-1]]  # reorder/garble
            if ref["authors"]:
                ref["authors"][0] = "Mismatch, X."
        elif field == "year":
            ref["year"] = entry["year"] + rng.choice([-2, -1, 1, 2, 3])
        else:  # title
            ref["title"] = entry["title"] + " (revised)"
        gold = {
            "reference_id": ref_id,
            "issue_kind": "metadata_mismatch",
            "flagged": True,
            "mismatch_field": field,
            "canonical": {k: entry[k] for k in ("authors", "year", "title", "doi")},
        }
        return ref, [], gold

    if kind == "dangling_uncited":
        # Listed in the bibliography but never cited in the body.
        ref["cited_in_text"] = False
        gold = {"reference_id": ref_id, "issue_kind": "dangling_uncited", "flagged": True}
        return ref, [], gold

    if kind == "dangling_missing":
        # An in-text marker that has NO matching reference. We emit no reference;
        # the gold record is keyed by the dangling marker id itself.
        marker = ref_id  # the marker id players must flag as missing
        gold = {"reference_id": marker, "issue_kind": "dangling_missing", "flagged": True}
        return None, [marker], gold

    if kind == "replication_uncited":
        # An original whose KNOWN replication is absent from the reference list.
        gold = {
            "reference_id": ref_id,
            "issue_kind": "replication_uncited",
            "flagged": True,
            "missing_replication": entry["known_replication"],
        }
        return ref, [], gold

    if kind == "miscitation":
        # The reference AS LISTED carries a known-wrong attribution (year/title).
        ref["year"] = entry["miscite_year"]
        ref["title"] = entry["miscite_title"]
        gold = {
            "reference_id": ref_id,
            "issue_kind": "miscitation",
            "flagged": True,
            "canonical": {"year": entry["year"], "title": entry["title"]},
        }
        return ref, [], gold

    if kind == "invalid_doi":
        # The DOI as listed is structurally malformed / non-resolvable. The arena
        # injected the corruption, so it stores the canonical DOI in gold.
        ref["doi"] = _corrupt_doi(entry["doi"], rng)
        gold = {
            "reference_id": ref_id,
            "issue_kind": "invalid_doi",
            "flagged": True,
            "canonical": {"doi": entry["doi"]},
        }
        return ref, [], gold

    if kind == "predatory_source":
        # Published in a known predatory / hijacked-style outlet. The arena swaps
        # the reputable canonical venue for the predatory twin it stored.
        ref["venue"] = entry["predatory_venue"]
        gold = {
            "reference_id": ref_id,
            "issue_kind": "predatory_source",
            "flagged": True,
            "canonical": {"venue": entry["venue"]},
        }
        return ref, [], gold

    if kind == "tortured_phrase":
        # The title carries a paper-mill tortured phrase (synonym-swapped
        # paraphrase). The canonical (legitimate) title is stored in gold.
        ref["title"] = entry["tortured_title"]
        gold = {
            "reference_id": ref_id,
            "issue_kind": "tortured_phrase",
            "flagged": True,
            "canonical": {"title": entry["title"]},
        }
        return ref, [], gold

    raise ValueError(f"unknown issue kind {kind!r}")


def _render_clean_control(pool_entry: dict, ref_id: str):
    """Render a CLEAN look-alike reference (flagged=False) that LOOKS suspicious.

    These come from catalogs/references.yaml#clean_controls and are the honest
    twins of the new mistake kinds (valid-but-unusual DOI, legitimate-but-obscure
    venue, real domain jargon). A correct player leaves them unflagged.
    """
    ref = _canonical_record(pool_entry, ref_id)
    ref["cited_in_text"] = True
    gold = {"reference_id": ref_id, "issue_kind": None, "flagged": False}
    return ref, [], gold


def _assemble(task_id, tier, plan, rng, split, visibility, catalog, controls) -> tuple[dict, dict]:
    """`plan` is a list of slot specs. Each slot is one of:
      - None / "clean"        : a clean reference picked from the main catalog.
      - "<issue_kind>"        : inject that issue (from ISSUE_KINDS).
      - "control:<pool>"      : a CLEAN look-alike from clean_controls[<pool>]
                                (the confusable honest twin of a mistake kind).
    """
    references: list[dict] = []
    in_text_marker_ids: list[str] = []
    gold_records: list[dict] = []
    used_ids: set = set()
    used_control_ids: set = set()

    for slot, spec in enumerate(plan):
        if isinstance(spec, str) and spec.startswith("control:"):
            pool_name = spec.split(":", 1)[1]
            pool = controls.get(pool_name) or []
            avail = [c for c in pool if c["id"] not in used_control_ids] or pool
            entry = rng.choice(avail)
            used_control_ids.add(entry["id"])
            ref_id = f"{entry['id']}__r{slot}"
            ref, extra_markers, gold = _render_clean_control(entry, ref_id)
        else:
            norm_kind = None if spec in (None, "clean") else spec
            entry = _pick_entry(catalog, rng, norm_kind or "clean", used_ids)
            used_ids.add(entry["id"])
            ref_id = f"{entry['id']}__r{slot}"
            ref, extra_markers, gold = _render_reference(entry, ref_id, norm_kind, rng)
        if ref is not None:
            references.append(ref)
            if ref.get("cited_in_text", True):
                in_text_marker_ids.append(ref["reference_id"])
        for m in extra_markers:
            in_text_marker_ids.append(m)
        gold_records.append(gold)

    # Shuffle the visible orderings so position carries no signal.
    rng.shuffle(references)
    rng.shuffle(in_text_marker_ids)

    mistake_kinds = sorted({g["issue_kind"] for g in gold_records if g["flagged"]})
    envelope = {
        "task_id": task_id,
        "arena_id": "reference-integrity-v1",
        "task_set_version": "v1",
        "split": split,
        "visibility": visibility,
        "difficulty": {"tier": tier, "n_issues": sum(1 for g in gold_records if g["flagged"])},
        "input": {
            "references": references,
            "in_text_marker_ids": in_text_marker_ids,
        },
    }
    gold = {"references": gold_records, "mistake_kinds": mistake_kinds}
    return envelope, gold


def generate(task_set_version: str, seed: int, split: str = "revealed"):
    visibility = "public" if split == "revealed" else "held_out"
    catalog = _load_catalog()
    controls = _load_clean_controls()
    control_specs = [f"control:{p}" for p in _CONTROL_POOLS]
    n_kinds = len(ISSUE_KINDS)
    base = 6  # references per task (clean tasks too)

    def emit(tier, idx, plan):
        tid = f"ri-t{tier}-{idx}-s{seed}"
        rng = random.Random(_seed_int(task_set_version, seed, tier, idx))
        env, gt = _assemble(tid, tier, plan, rng, split, visibility, catalog, controls)
        _GROUND_TRUTH_CACHE[tid] = gt
        return env

    # T1: clean/simple — every reference well-formed, cited, no issues.
    for k in range(3):
        yield emit(1, k, [None] * base)

    # T2: controls-only / false-alarm trap. References that LOOK suspicious
    # (reordered author initials, DOI suffixes, an uncited-looking-but-cited ref,
    # AND the confusable honest twins of the new kinds — a valid-but-unusual DOI,
    # a legitimate-but-obscure venue, real domain jargon) but are ALL CLEAN. A
    # good player must NOT flag any of them. n_issues == 0.
    for k in range(3):
        plan: list = [None] * base
        # Seat every clean-control look-alike pool deterministically; the rest of
        # the slots are ordinary clean references. Offset by k so the controls
        # land in different positions across the three T2 tasks (still seed-free).
        for j, spec in enumerate(control_specs):
            plan[(k + j) % base] = spec
        yield emit(2, k, plan)

    # T3: exactly one injected issue, cycling through EVERY issue kind (so every
    # kind appears at least once in the revealed set), the rest clean.
    for i, kind in enumerate(ISSUE_KINDS):
        plan = [None] * base
        plan[i % base] = kind
        yield emit(3, i, plan)

    # T4: a single SUBTLE issue embedded right next to its confusable CLEAN twin
    # (the realism tier). Each subtle kind is paired with the look-alike pool a
    # naive checker would confuse it with, so the player must discriminate.
    subtle_pairs = [
        ("metadata_mismatch", None),                 # vs an ordinary clean ref
        ("miscitation", None),
        ("invalid_doi", "control:legit_unusual_doi"),
        ("predatory_source", "control:obscure_venue"),
        ("tortured_phrase", "control:legit_jargon_title"),
    ]
    for i, (kind, twin) in enumerate(subtle_pairs):
        plan = [None] * base
        plan[(i + 1) % base] = kind
        if twin is not None:
            plan[(i + 2) % base] = twin  # the honest look-alike, side by side
        yield emit(4, i, plan)

    # T5: MULTIPLE co-occurring issues (deterministic cycling offset across kinds),
    # the remaining slots clean.
    for k in range(3):
        plan = [None] * base
        for o in range(3):
            slot = (k + o) % base
            plan[slot] = ISSUE_KINDS[(k * 3 + o) % n_kinds]
        yield emit(5, k, plan)

    # T6: full composition — a deterministic mix spanning every issue kind across
    # the slots (the slot count is `base`; the kind list is longer, so each T6
    # task selects a rotating window — across the two tasks every kind appears).
    for k in range(2):
        plan = [ISSUE_KINDS[(j + k * base) % n_kinds] for j in range(base)]
        yield emit(6, k, plan)


def ground_truth(task_id: str) -> dict:
    """Return gold for a task. Regenerated from seed via the in-process cache.

    The runner always calls generate() before ground_truth(); this arena needs no
    external gold registry because the secret is the private seed, not a stored
    answer key.
    """
    if task_id not in _GROUND_TRUTH_CACHE:
        raise KeyError(
            f"No cached gold for {task_id!r}; call generate() for the matching "
            "split/seed before ground_truth()."
        )
    return _GROUND_TRUTH_CACHE[task_id]
