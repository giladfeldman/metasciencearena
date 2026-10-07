"""Generator for significance-language-v1 (metacheck-style spin/marginal arena).

Builds manuscript excerpts (`{text}`) with KNOWN interpretive-language problems
injected as flagged spans, mirroring metacheck's `marginal` and `causal_claims`
modules. All-procedural and deterministic from (task_set_version, seed).

Three injected mistake KINDS (== mistake_kinds, cycled deterministically by index,
NOT rng.choice — so every split covers every kind equally):
  marginal_significance : "marginally significant" / "trend toward" /
                          "approaching significance" for p in (.05, .10).
  spin_overclaim        : a strong claim not supported by the reported result.
  causal_overclaim      : causal language without a randomised design.

Clean controls (the T2 false-alarm trap) interleave legitimate hedging, exact
significant claims, and causal claims that DO cite randomisation — a good player
must NOT flag them.

Dual-benchmark (revealed/private): both splits run the IDENTICAL tier matrix and
assign mistake KINDS deterministically (index-driven, seed-independent), so every
split covers the full array of injected mistakes equally — this is what makes
framework/parity.py pass. Only the concrete sentence WORDING (which span_option /
which clean flavour) is seed-driven, so revealed and private content still differ.

Gold is regenerated from the seed and served from the in-process cache (no external
registry): the secret is the private seed, not a stored answer key. The revealed
seed is committed in arena.yaml#benchmark_splits.
"""
from __future__ import annotations

import hashlib
import random
from pathlib import Path

import yaml

ARENA_DIR = Path(__file__).resolve().parent
CATALOGS_DIR = ARENA_DIR / "catalogs"

_GROUND_TRUTH_CACHE: dict[str, dict] = {}

# The injected mistake kinds, in the deterministic cycling order. Index
# position is what drives which kind a slot gets (NOT rng.choice).
#   marginal_significance / spin_overclaim / causal_overclaim — original three.
#   one_sided_unjustified — switching to a one-tailed test post hoc only because it
#                           crosses .05 (vs a PRE-REGISTERED directional test, which
#                           is legitimate — clean_prereg_onesided).
#   p_just_over_threshold_spin — a p just above .05 (.053/.058/.061) asserted as a
#                           significant effect (vs honestly reporting p>.05 as n.s. —
#                           clean_hedge).
MISTAKE_KINDS = [
    "marginal_significance",
    "spin_overclaim",
    "causal_overclaim",
    "one_sided_unjustified",
    "p_just_over_threshold_spin",
]

# Clean-control flavours used to fill non-mistake slots and the whole T2 trap.
# clean_prereg_onesided is the look-alike for one_sided_unjustified; p_just_over_
# threshold_spin's look-alike is clean_hedge (an honest p>.05 report).
CLEAN_FLAVOURS = ["clean_hedge", "clean_exact", "clean_causal", "clean_prereg_onesided"]


def _load_catalog() -> dict:
    with (CATALOGS_DIR / "sentences.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _seed_int(*parts) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def _render_sentence(catalog: dict, flavour: str, idx: int, rng: random.Random) -> tuple[str, str | None, str | None]:
    """Render one sentence for a given flavour.

    Returns (sentence_text, span_text, category). `span_text`/`category` are the
    phrase to flag and its category for a genuine mistake; both None for a clean
    control (nothing to flag).
    """
    entries = catalog[flavour]
    entry = entries[idx % len(entries)]
    span = rng.choice(entry["span_options"])
    sentence = entry["template"].format(span=span)
    if flavour in MISTAKE_KINDS:
        return sentence, span, entry["category"]
    return sentence, None, None


def _assemble(task_id, tier, slot_flavours, rng, split, visibility) -> tuple[dict, dict]:
    """Concatenate rendered sentences into one excerpt and compute gold spans.

    slot_flavours: list of flavour strings (mistake kinds and/or clean_* controls),
    one per sentence slot, in order. Char offsets of each flagged span are computed
    against the final concatenated text.
    """
    parts: list[str] = []
    flags: list[dict] = []
    mistake_kinds: list[str] = []
    cursor = 0
    for i, flavour in enumerate(slot_flavours):
        sentence, span, category = _render_sentence(catalog=_CATALOG, flavour=flavour, idx=i, rng=rng)
        # Position of this sentence in the assembled text (sentences joined by " ").
        start_of_sentence = cursor
        parts.append(sentence)
        if span is not None and category is not None:
            # char offset of the span within the full text == sentence start +
            # offset of span within the sentence (templates contain it verbatim once).
            local = sentence.index(span)
            char_start = start_of_sentence + local
            char_end = char_start + len(span)
            flags.append({
                "span": {"text": span, "char_start": char_start, "char_end": char_end},
                "category": category,
            })
            mistake_kinds.append(category)
        # advance cursor past this sentence plus the joining space.
        cursor += len(sentence) + 1

    text = " ".join(parts)
    n_flags = len(flags)
    envelope = {
        "task_id": task_id,
        "arena_id": "significance-language-v1",
        "task_set_version": "v1",
        "split": split,
        "visibility": visibility,
        "difficulty": {"tier": tier, "n_flags": n_flags},
        "input": {"text": text},
    }
    gold = {"flags": flags, "mistake_kinds": mistake_kinds}
    return envelope, gold


# Catalog is loaded once at import; sentence selection within it is seed-driven.
_CATALOG = _load_catalog()


def generate(task_set_version: str, seed: int, split: str = "revealed"):
    visibility = "public" if split == "revealed" else "held_out"
    kinds = MISTAKE_KINDS
    n_kinds = len(kinds)

    def emit(tier, idx, slot_flavours):
        tid = f"sl-t{tier}-{idx}-s{seed}"
        rng = random.Random(_seed_int(task_set_version, seed, tier, idx))
        env, gt = _assemble(tid, tier, slot_flavours, rng, split, visibility)
        _GROUND_TRUTH_CACHE[tid] = gt
        return env

    # T1: clean/simple — short excerpts, all clean controls (one clean flavour each).
    # No flags. Establishes the floor: a quiet, correctly-written passage.
    for k in range(3):
        yield emit(1, k, [CLEAN_FLAVOURS[k % len(CLEAN_FLAVOURS)]])

    # T2: false-alarm TRAP — excerpts built ENTIRELY from clean controls that look
    # suspicious (legit hedging, exact-significant claims, randomised causal claims).
    # n_flags must be 0; a good player flags NOTHING here. Cycle all clean flavours.
    for k in range(3):
        flavours = [CLEAN_FLAVOURS[(k + j) % len(CLEAN_FLAVOURS)] for j in range(3)]
        yield emit(2, k, flavours)

    # T3: exactly ONE injected mistake (cycling through every kind so each kind is
    # covered), surrounded by clean controls.
    for i in range(n_kinds):
        flavours = [kinds[i], CLEAN_FLAVOURS[i % len(CLEAN_FLAVOURS)], CLEAN_FLAVOURS[(i + 1) % len(CLEAN_FLAVOURS)]]
        yield emit(3, i, flavours)

    # T4: subtle — one injected mistake of each kind, but embedded amid a matched
    # clean control of a confusable flavour (e.g. a marginal mistake next to a
    # legitimate hedge; a causal overclaim next to a randomised causal claim).
    # Pairing the mistake with its look-alike clean control is what makes it subtle.
    _CONFUSABLE = {
        "marginal_significance": "clean_hedge",
        "spin_overclaim": "clean_exact",
        "causal_overclaim": "clean_causal",
        # one-sided-unjustified next to a legitimate PRE-REGISTERED one-sided test;
        # p-just-over-threshold spin next to an honest p>.05 non-significant report.
        "one_sided_unjustified": "clean_prereg_onesided",
        "p_just_over_threshold_spin": "clean_hedge",
    }
    for i in range(n_kinds):
        kind = kinds[i]
        flavours = [_CONFUSABLE[kind], kind, _CONFUSABLE[kind]]
        yield emit(4, i, flavours)

    # T5: MULTIPLE injected mistakes co-occurring — all three kinds in one excerpt,
    # interleaved with clean controls.
    for k in range(3):
        flavours = [
            kinds[(k + 0) % n_kinds],
            CLEAN_FLAVOURS[k % len(CLEAN_FLAVOURS)],
            kinds[(k + 1) % n_kinds],
            kinds[(k + 2) % n_kinds],
        ]
        yield emit(5, k, flavours)

    # T6: full composition — a deterministic mix of every mistake kind and every
    # clean flavour across many slots (the hardest discrimination).
    for k in range(2):
        flavours = []
        for j in range(6):
            if (j + k) % 2 == 0:
                flavours.append(kinds[(j + k) % n_kinds])
            else:
                flavours.append(CLEAN_FLAVOURS[(j + k) % len(CLEAN_FLAVOURS)])
        yield emit(6, k, flavours)


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
