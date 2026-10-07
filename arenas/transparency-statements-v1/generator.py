"""Generator for transparency-statements-v1 (metacheck-style field-map arena).

Builds plain-text manuscript transparency sections and the TRUE field map a
player must extract: per-field present/absent, on_request hedges, and whether a
quoted repository URL is a REAL repo or a placeholder/broken link. Mirrors
metacheck's coi_check, funding_check, open_practices and all_urls modules.
All-procedural and deterministic from (task_set_version, seed).

Dual-benchmark (revealed/private): both splits run the IDENTICAL tier matrix and
assign the injected-mistake KIND deterministically (index-driven cycling through
ALL kinds, seed-independent), so every split covers the full array of injected
mistakes equally — this is what makes framework/parity.py pass. Only the concrete
statement TEXT/URLs are seed-driven, so revealed and private content still differ.

Gold is regenerated from the seed and served from the in-process cache (no
external registry needed): the secret is the private seed, not a stored answer
key. The revealed seed is committed in arena.yaml#benchmark_splits.

Injected mistake kinds (cycled deterministically):
  - missing_coi               : the COI statement is omitted entirely.
  - missing_funding           : the funding statement is omitted entirely.
  - data_on_request_not_real  : an open-practices field claims "available on
                                request" instead of giving a real repository link.
  - placeholder_url           : an open-practices field presents a fake/broken URL
                                as if it were a real repository.
  - false_open_claim          : an open-practices field ASSERTS openness ("All data
                                are openly available.") but quotes NO link at all —
                                an unverifiable open-data/-code claim. Gold marks
                                available=False, on_request=False, url=None.
  - false_prereg_claim        : the prereg field claims the study "was
                                preregistered" but gives NO registry link/ID. Gold
                                marks available=False, url=None.
  - funding_on_request        : the funding line defers disclosure ("funding details
                                available on request") instead of naming the source —
                                not an actual funding statement. Gold marks
                                present=False.
Clean variant: real COI + funding + real repository URLs for data/code/materials/
prereg. The T2 paraphrase tier reworders every field but injects NO mistake; it
also carries the confusable clean controls — a genuine "no external funding"
declaration (present, the look-alike for funding_on_request) and openness/prereg
claims WITH a real link (the look-alikes for false_open_claim / false_prereg_claim).
"""
from __future__ import annotations

import hashlib
import random
from pathlib import Path

import yaml

ARENA_DIR = Path(__file__).resolve().parent
CATALOGS_DIR = ARENA_DIR / "catalogs"

_GROUND_TRUTH_CACHE: dict[str, dict] = {}

ARENA_ID = "transparency-statements-v1"

# The closed set of injected-mistake kinds, in deterministic cycling order.
MISTAKE_KINDS = [
    "missing_coi",
    "missing_funding",
    "data_on_request_not_real",
    "placeholder_url",
    "false_open_claim",
    "false_prereg_claim",
    "funding_on_request",
]

# The four open-practices fields (in order) that a URL mistake can target.
OPEN_FIELDS = ["data", "code", "materials", "prereg"]

# The three open-practices fields a bare "openly available" (no-link) claim can
# target; prereg's no-link claim is its own kind (false_prereg_claim).
BARE_OPEN_FIELDS = ["data", "code", "materials"]

# Mistake kinds whose target open-practices field CYCLES positionally (the idx
# passed to _apply_mistake selects which field). The other open-field kind,
# false_prereg_claim, always targets prereg and ignores idx, so it is excluded.
_CYCLING_TARGET_MISTAKES = {
    "data_on_request_not_real",
    "placeholder_url",
    "false_open_claim",
}


def _load_catalog() -> dict:
    with (CATALOGS_DIR / "statements.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _seed_int(*parts) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def _blank_field_gold(field: str) -> dict:
    """Default (absent) gold for one field."""
    if field in ("coi", "funding"):
        return {"present": False, "statement": ""}
    if field == "prereg":
        return {"available": False, "url": None}
    return {"available": False, "on_request": False, "url": None}


# --- per-field renderers: each returns (sentence_or_None, field_gold) ---------

def _render_coi(cat: dict, rng: random.Random, mode: str) -> tuple[str | None, dict]:
    if mode == "missing":
        return None, {"present": False, "statement": ""}
    pool = cat["coi"]["paraphrase_options"] if mode == "paraphrase" else cat["coi"]["real_options"]
    s = rng.choice(pool)
    return s, {"present": True, "statement": s}


def _render_funding(cat: dict, rng: random.Random, mode: str) -> tuple[str | None, dict]:
    if mode == "missing":
        return None, {"present": False, "statement": ""}
    f = cat["funding"]
    if mode == "on_request":
        # funding_on_request MISTAKE: a funding line that defers disclosure behind
        # a request instead of naming a source — NOT an actual funding statement.
        s = rng.choice(f["on_request_options"])
        return s, {"present": False, "statement": ""}
    if mode == "no_funding":
        # CLEAN CONTROL: a genuine no-external-funding declaration is a complete
        # disclosure (present=True), the confusable look-alike for funding_on_request.
        s = rng.choice(f["no_funding_options"])
        return s, {"present": True, "statement": s}
    grant = rng.choice(f["grants"])
    agency = rng.choice(f["agencies"])
    pool = f["paraphrase_options"] if mode == "paraphrase" else f["real_options"]
    s = rng.choice(pool).format(grant=grant, agency=agency)
    return s, {"present": True, "statement": s}


def _render_open(cat: dict, field: str, rng: random.Random, mode: str) -> tuple[str | None, dict]:
    """Render one open-practices field (data/code/materials/prereg).

    Modes:
      present | paraphrase  -> real repo URL: available=True, on_request=False.
      on_request            -> "available on request": available=False,
                               on_request=True, url=None. (data_on_request_not_real)
      placeholder           -> fake/broken URL presented as a repo: the manuscript
                               claims availability but the link is NOT real, so gold
                               marks available=False with a non-null url. (placeholder_url)
      bare_open             -> data/code/materials: ASSERTS openness ("openly
                               available in a public repository") but quotes NO link.
                               Gold available=False, on_request=False, url=None — the
                               assertion is not a verifiable real repo. (false_open_claim)
      bare_prereg           -> prereg: claims "was preregistered" with NO link/ID.
                               Gold available=False, url=None. (false_prereg_claim)
      missing               -> field omitted entirely (absent).
    """
    spec = cat["open_practices"][field]

    if mode == "missing":
        return None, _blank_field_gold(field)

    if mode == "on_request":
        s = spec["on_request_template"]
        if field == "prereg":
            return s, {"available": False, "url": None}
        return s, {"available": False, "on_request": True, "url": None}

    if mode == "bare_open":
        # false_open_claim: an openness assertion with no link. Only data/code/
        # materials use this; prereg's no-link claim is bare_prereg.
        s = spec["bare_open_template"]
        return s, {"available": False, "on_request": False, "url": None}

    if mode == "bare_prereg":
        # false_prereg_claim: "was preregistered" with no registry link/ID.
        s = spec["bare_prereg_template"]
        return s, {"available": False, "url": None}

    if mode == "placeholder":
        url = rng.choice(spec["placeholder_url_options"])
        s = spec["placeholder_template"].format(url=url)
        if field == "prereg":
            return s, {"available": False, "url": url}
        return s, {"available": False, "on_request": False, "url": url}

    # present / paraphrase: a real repository link.
    url = rng.choice(spec["real_url_options"])
    template = spec["paraphrase_template"] if mode == "paraphrase" else spec["present_template"]
    s = template.format(url=url)
    if field == "prereg":
        return s, {"available": True, "url": url}
    return s, {"available": True, "on_request": False, "url": url}


def _assemble(task_id, tier, field_modes, injected_kinds, rng, split, visibility):
    """Render every field, collect gold + the manuscript text.

    field_modes: dict field -> mode string.
    injected_kinds: the deterministic list of mistake-kind labels for this task.
    """
    cat = _load_catalog()
    gold_fields: dict[str, dict] = {}
    lines: list[str] = []

    # Order the section the way real manuscripts do.
    order = ["coi", "funding", "data", "code", "materials", "prereg"]
    headers = {
        "coi": "Competing interests.",
        "funding": "Funding.",
        "data": "Data availability.",
        "code": "Code availability.",
        "materials": "Materials availability.",
        "prereg": "Preregistration.",
    }
    for field in order:
        mode = field_modes[field]
        if field == "coi":
            s, g = _render_coi(cat, rng, mode)
        elif field == "funding":
            s, g = _render_funding(cat, rng, mode)
        else:
            s, g = _render_open(cat, field, rng, mode)
        gold_fields[field] = g
        if s is not None:
            lines.append(f"{headers[field]} {s}")

    n_mistakes = len(injected_kinds)
    envelope = {
        "task_id": task_id,
        "arena_id": ARENA_ID,
        "task_set_version": "v1",
        "split": split,
        "visibility": visibility,
        "difficulty": {"tier": tier, "n_mistakes": n_mistakes},
        "input": {"text": "\n".join(lines)},
    }
    gold = dict(gold_fields)
    gold["mistake_kinds"] = list(injected_kinds)
    return envelope, gold


# --- mistake -> field-mode plan ----------------------------------------------

def _apply_mistake(field_modes: dict, kind: str, idx: int) -> None:
    """Mutate field_modes in place to inject one mistake of `kind`.

    For open-practice mistakes the target field cycles deterministically through
    OPEN_FIELDS (or BARE_OPEN_FIELDS for false_open_claim) by idx, so across a tier
    every applicable field gets exercised. false_prereg_claim always targets prereg
    and funding_on_request always targets funding.
    """
    if kind == "missing_coi":
        field_modes["coi"] = "missing"
    elif kind == "missing_funding":
        field_modes["funding"] = "missing"
    elif kind == "data_on_request_not_real":
        target = OPEN_FIELDS[idx % len(OPEN_FIELDS)]
        field_modes[target] = "on_request"
    elif kind == "placeholder_url":
        target = OPEN_FIELDS[idx % len(OPEN_FIELDS)]
        field_modes[target] = "placeholder"
    elif kind == "false_open_claim":
        target = BARE_OPEN_FIELDS[idx % len(BARE_OPEN_FIELDS)]
        field_modes[target] = "bare_open"
    elif kind == "false_prereg_claim":
        field_modes["prereg"] = "bare_prereg"
    elif kind == "funding_on_request":
        field_modes["funding"] = "on_request"
    else:  # pragma: no cover - guard
        raise ValueError(f"unknown mistake kind {kind!r}")


def _target_idx(kind: str, target_field: str | None, fallback: int) -> int:
    """Resolve the idx `_apply_mistake` needs to land `kind` on `target_field`.

    For open-field mistakes that cycle a target (data_on_request_not_real,
    placeholder_url, false_open_claim) a named `target_field` is converted to its
    position in the appropriate pool; everything else (missing_*, funding_on_request,
    false_prereg_claim — which ignore idx) uses the `fallback`.
    """
    if target_field is None:
        return fallback
    if kind not in _CYCLING_TARGET_MISTAKES:  # pragma: no cover - misuse guard
        raise ValueError(
            f"kind {kind!r} does not take a positional target_field "
            f"(got {target_field!r}); pass None and let it use its fixed field."
        )
    pool = BARE_OPEN_FIELDS if kind == "false_open_claim" else OPEN_FIELDS
    return pool.index(target_field)


def _all_clean(paraphrase: bool = False) -> dict:
    mode = "paraphrase" if paraphrase else "present"
    return {f: mode for f in ["coi", "funding", "data", "code", "materials", "prereg"]}


def generate(task_set_version: str, seed: int, split: str = "revealed"):
    visibility = "public" if split == "revealed" else "held_out"

    def emit(tier, idx, field_modes, injected_kinds):
        tid = f"ts-t{tier}-{idx}-s{seed}"
        rng = random.Random(_seed_int(task_set_version, seed, tier, idx))
        env, gt = _assemble(tid, tier, field_modes, injected_kinds, rng, split, visibility)
        _GROUND_TRUTH_CACHE[tid] = gt
        return env

    # T1: clean/simple — every field a real, well-formed statement. No mistakes.
    for k in range(3):
        yield emit(1, k, _all_clean(paraphrase=False), [])

    # T2: paraphrase-consistent FALSE-ALARM TRAP — every field reworded but still
    # a real, correct statement. A good player must NOT flag any field. The third
    # task swaps funding to a genuine "no external funding" declaration: a CLEAN
    # CONTROL (present=True) that is the confusable look-alike for the
    # funding_on_request mistake (both lack a grant/agency) and must NOT be flagged.
    for k in range(3):
        fm = _all_clean(paraphrase=False) | _all_clean(paraphrase=True)
        if k == 2:
            fm["funding"] = "no_funding"
        yield emit(2, k, fm, [])

    # T3: single injected mistake, cycling through ALL mistake kinds so every kind
    # is covered. Rest of the fields are clean (verbatim).
    for i, kind in enumerate(MISTAKE_KINDS):
        fm = _all_clean(paraphrase=False)
        _apply_mistake(fm, kind, i)
        yield emit(3, i, fm, [kind])

    # T4: subtle mistake — a single mistake hidden amid PARAPHRASED (reworded) clean
    # fields, so the rewording noise makes the one real mistake harder to spot. An
    # explicit, index-driven (kind, target_field) matrix (seed-independent structure
    # → parity holds): each open-practices mistake is paired with its confusable
    # clean look-alike (a real link / a complete disclosure) on the other fields.
    t4_cases: list[tuple[str, str | None]] = []
    for f in OPEN_FIELDS:                       # data_on_request_not_real on every open field
        t4_cases.append(("data_on_request_not_real", f))
    for f in OPEN_FIELDS:                       # placeholder_url on every open field
        t4_cases.append(("placeholder_url", f))
    for f in BARE_OPEN_FIELDS:                  # false_open_claim on data/code/materials
        t4_cases.append(("false_open_claim", f))
    t4_cases.append(("false_prereg_claim", None))   # bare "was preregistered", no link (prereg)
    t4_cases.append(("funding_on_request", None))   # funding deferred behind a request
    for t4_idx, (kind, target_field) in enumerate(t4_cases):
        fm = _all_clean(paraphrase=True)
        _apply_mistake(fm, kind, _target_idx(kind, target_field, t4_idx))
        yield emit(4, t4_idx, fm, [kind])

    # T5: multiple co-occurring mistakes amid paraphrased others. An explicit set of
    # field-disjoint pairs (so the two mistakes never overwrite each other), covering
    # the legacy pairs plus the newly added kinds.
    t5_cases: list[list[tuple[str, str | None]]] = [
        [("missing_coi", None), ("missing_funding", None)],
        [("missing_funding", None), ("data_on_request_not_real", "materials")],
        [("data_on_request_not_real", "materials"), ("placeholder_url", "prereg")],
        [("false_open_claim", "data"), ("funding_on_request", None)],
        [("false_prereg_claim", None), ("placeholder_url", "data")],
    ]
    for k, pairs in enumerate(t5_cases):
        fm = _all_clean(paraphrase=True)
        injected = []
        for kind, target_field in pairs:
            _apply_mistake(fm, kind, _target_idx(kind, target_field, k))
            injected.append(kind)
        yield emit(5, k, fm, injected)

    # T6: full composition — a deterministic mix spanning ALL mistake kinds across
    # the two tasks (their union covers every kind), over fields that are otherwise
    # clean/paraphrased. Each task's mistakes target DISJOINT fields so no kind
    # overwrites another (missing_funding and funding_on_request both target funding,
    # so they live in different tasks).
    t6_cases: list[list[tuple[str, str | None]]] = [
        [("missing_coi", None), ("missing_funding", None),
         ("data_on_request_not_real", "data"), ("placeholder_url", "code"),
         ("false_prereg_claim", None)],
        [("funding_on_request", None), ("false_open_claim", "data"),
         ("data_on_request_not_real", "code"), ("false_prereg_claim", None)],
    ]
    for k, items in enumerate(t6_cases):
        fm = _all_clean(paraphrase=(k == 1))
        injected = []
        for kind, target_field in items:
            _apply_mistake(fm, kind, _target_idx(kind, target_field, k))
            injected.append(kind)
        yield emit(6, k, fm, injected)


def ground_truth(task_id: str) -> dict:
    """Return gold for a task. Regenerated from seed via the in-process cache.

    The runner always calls generate() before ground_truth(); this arena needs no
    external gold registry because the secret is the private seed, not a stored
    answer key. Raises KeyError if the task_id has not been generated.
    """
    if task_id not in _GROUND_TRUTH_CACHE:
        raise KeyError(
            f"No cached gold for {task_id!r}; call generate() for the matching "
            "split/seed before ground_truth()."
        )
    return _GROUND_TRUTH_CACHE[task_id]
