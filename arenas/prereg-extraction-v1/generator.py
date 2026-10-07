"""Generator for prereg-extraction-v1 (field-map arena).

Builds paper-text tasks that may embed a preregistration link (OSF or
AsPredicted) plus its registration content. The player must (a) detect whether a
prereg is present, (b) name the platform, (c) recover the canonical link, and
(d) field-map the registration into {hypotheses, design, sample_size,
analysis_plan}. All-procedural and deterministic from (task_set_version, seed).

Dual-benchmark (revealed/private): both splits run the IDENTICAL tier matrix and
assign the injected-mistake KIND deterministically (index-driven cycling through
ALL kinds, NOT rng.choice), so every split covers the full array of injected
mistakes equally — this is what makes framework/parity.py pass. Only the concrete
surrounding narrative WORDING is seed-driven, so revealed and private content
still differ.

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

# The INJECTED mistake kinds, cycled deterministically so every split covers all
# of them. (Clean controls carry no injected mistake.)
#
# The first three are the original field-map mistakes. The last three are real
# OSF preregistration integrity failures mined from the AbusingPreReg taxonomy
# (Tier D modules 12/13/17), each with a confusable CLEAN look-alike that must
# NOT be flagged:
#   - viewonly_instead_of_doi   (module 13): the paper links an anonymized
#       view-only OSF URL (osf.io/<frag>/?view_only=<token>) instead of the
#       canonical registration link. The registration IS real and its content is
#       readable, so prereg_found=True and the fields are extractable; the trap is
#       recovering the canonical osf.io link and still naming the platform "osf".
#       Clean look-alike: a correct registration DOI link
#       (doi.org/10.17605/OSF.IO/<FRAG>) — also OSF, also found, fields present.
#   - embargoed_at_publication  (module 12): the paper is public while its
#       registration is still under embargo, so the registration is REFERENCED but
#       its content is not accessible — prereg_found=True (a registration exists)
#       yet the four fields are unrecoverable (null). Clean look-alike: an embargo
#       that LIFTED before publication, so the same registration's fields ARE
#       readable and must be extracted.
#   - withdrawn_still_cited      (module 17): the paper still cites a registration
#       that has been WITHDRAWN, leaving only a tombstone — the plan no longer
#       exists, so there is no usable preregistration: prereg_found=False even
#       though the text claims "pre-registered" and shows a (dead) link. Clean
#       look-alike: a live registration cited correctly (found=True, fields present).
INJECTED_KINDS = [
    "no_prereg",
    "prereg_plaintext",
    "wrong_platform_schema",
    "viewonly_instead_of_doi",
    "embargoed_at_publication",
    "withdrawn_still_cited",
]

# Field ids the player must extract.
FIELDS = ["hypotheses", "design", "sample_size", "analysis_plan"]

# Narrative framings that vary only the surrounding wording (seed-driven), so
# revealed vs private text differs without touching the gold field values.
_INTRO_OPTIONS = [
    "This study investigated {topic}.",
    "We report a study on {topic}.",
    "The present work examined {topic}.",
    "Here we present an investigation of {topic}.",
]
# Decoy sentences for the T2 false-alarm trap: they TALK about preregistration
# but embed no actual registration link or content.
_DECOY_OPTIONS = [
    "We are strong advocates of preregistration and open science, though "
    "constraints meant this particular study was not preregistered.",
    "Preregistration is best practice in our field; future work from our lab "
    "will be registered on the Open Science Framework.",
    "While we did not preregister, all materials follow open-science norms and "
    "AsPredicted-style transparency.",
    "This exploratory study was not preregistered; we note OSF and AsPredicted "
    "as registries we intend to use going forward.",
]


def _load_studies() -> list[dict]:
    with (CATALOGS_DIR / "studies.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _seed_int(*parts) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def _osf_link(study: dict) -> str:
    return f"https://osf.io/{study['osf_frag']}/"


def _aspredicted_link(study: dict) -> str:
    return f"https://aspredicted.org/{study['aspredicted_frag']}.pdf"


def _osf_doi_link(study: dict) -> str:
    """Canonical OSF registration DOI link (the authoritative, frozen form)."""
    return f"https://doi.org/10.17605/OSF.IO/{study['osf_frag'].upper()}"


def _osf_viewonly_link(study: dict) -> str:
    """Anonymized OSF view-only link (mutable, non-canonical access form)."""
    return f"https://osf.io/{study['osf_frag']}/?view_only={study['view_only_token']}"


def _osf_block(study: dict, link: str) -> str:
    """Clean, structured OSF-style registration block."""
    return (
        f"Preregistration: this study was preregistered on the Open Science "
        f"Framework ({link}).\n"
        f"Hypotheses: {study['hypotheses']}\n"
        f"Design: {study['design']}\n"
        f"Sample size: {study['sample_size']}\n"
        f"Analysis plan: {study['analysis_plan']}"
    )


def _aspredicted_block(study: dict, link: str) -> str:
    """Clean, structured AsPredicted-style registration block (numbered Qs)."""
    return (
        f"This study was preregistered on AsPredicted ({link}).\n"
        f"1) Hypothesis: {study['hypotheses']}\n"
        f"2) Design plan: {study['design']}\n"
        f"3) Sample size: {study['sample_size']}\n"
        f"4) Analyses: {study['analysis_plan']}"
    )


def _plaintext_block(study: dict, link: str) -> str:
    """Unstructured plain-text registration: fields present but with no labels."""
    return (
        f"Our registration is available at {link}. "
        f"In it we predicted that {study['hypotheses'].rstrip('.').lower()}. "
        f"We planned {study['design'].rstrip('.').lower()}, recruiting "
        f"{study['sample_size'].rstrip('.').lower()}, and to test this with "
        f"{study['analysis_plan'].rstrip('.').lower()}."
    )


def _wrong_schema_block(study: dict, link: str, platform: str) -> str:
    """A platform/schema mismatch: AsPredicted numbered-question labels sitting
    under an OSF link (platform == 'osf'), or OSF-style prose labels under an
    AsPredicted link (platform == 'aspredicted'). The fields are still present
    and extractable; only the schema/platform pairing is wrong."""
    if platform == "osf":
        # OSF link, but AsPredicted's numbered-question schema.
        return (
            f"Preregistered on the Open Science Framework ({link}).\n"
            f"1) Hypothesis: {study['hypotheses']}\n"
            f"2) Design plan: {study['design']}\n"
            f"3) Sample size: {study['sample_size']}\n"
            f"4) Analyses: {study['analysis_plan']}"
        )
    # AsPredicted link, but OSF's prose-label schema.
    return (
        f"This study was preregistered on AsPredicted ({link}).\n"
        f"Hypotheses: {study['hypotheses']}\n"
        f"Design: {study['design']}\n"
        f"Sample size: {study['sample_size']}\n"
        f"Analysis plan: {study['analysis_plan']}"
    )


def _viewonly_block(study: dict, viewonly_link: str) -> str:
    """A real OSF registration linked only via an anonymized view-only URL.

    The registration content is fully readable (fields extractable) — the abuse
    is that the manuscript points at the mutable, anonymized view-only page rather
    than the canonical osf.io registration link (AbusingPreReg module 13)."""
    return (
        f"Preregistration: the registration for this study is available as an "
        f"anonymized link ({viewonly_link}).\n"
        f"Hypotheses: {study['hypotheses']}\n"
        f"Design: {study['design']}\n"
        f"Sample size: {study['sample_size']}\n"
        f"Analysis plan: {study['analysis_plan']}"
    )


def _doi_block(study: dict, doi_link: str) -> str:
    """Clean control for the view-only abuse: the canonical OSF registration DOI.

    This is the honest, frozen form — an osf.io registration cited by its DOI.
    A naive player may fail to recognize the doi.org URL as an OSF registration;
    a good player names the platform 'osf' and extracts every field."""
    return (
        f"Preregistration: this study was preregistered on the Open Science "
        f"Framework, registration DOI {doi_link}.\n"
        f"Hypotheses: {study['hypotheses']}\n"
        f"Design: {study['design']}\n"
        f"Sample size: {study['sample_size']}\n"
        f"Analysis plan: {study['analysis_plan']}"
    )


def _embargoed_block(study: dict, link: str) -> str:
    """A registration that exists but is still EMBARGOED at publication.

    The registration is referenced (so prereg_found=True) but its content is NOT
    public yet — the four fields are unrecoverable (AbusingPreReg module 12). A
    careful player reports the registration as present but leaves the fields null
    rather than inventing them."""
    return (
        f"Preregistration: this study was preregistered on the Open Science "
        f"Framework ({link}). The registration remains under embargo at the time "
        f"of publication, so its full contents are not yet publicly accessible; "
        f"the registration page currently shows only metadata."
    )


def _embargo_lifted_block(study: dict, link: str) -> str:
    """Clean control for the embargo abuse: the embargo LIFTED before publication.

    Same registration, but its content is now public — the fields ARE readable
    and must be extracted. The 'embargo' wording is the confusable look-alike that
    must NOT cause the player to give up on the fields."""
    return (
        f"Preregistration: this study was preregistered on the Open Science "
        f"Framework ({link}). The registration was initially embargoed; the "
        f"embargo lifted prior to publication and the full registration is now "
        f"public.\n"
        f"Hypotheses: {study['hypotheses']}\n"
        f"Design: {study['design']}\n"
        f"Sample size: {study['sample_size']}\n"
        f"Analysis plan: {study['analysis_plan']}"
    )


def _withdrawn_block(study: dict, link: str) -> str:
    """A WITHDRAWN registration still cited as pre-registered (tombstone).

    The manuscript claims pre-registration and shows a link, but the registration
    has been withdrawn — only a tombstone remains, so there is no usable
    preregistration: prereg_found=False (AbusingPreReg module 17). The trap is
    that the text asserts 'pre-registered' and carries a link."""
    return (
        f"This study was preregistered on the Open Science Framework ({link}). "
        f"Note: this registration has since been withdrawn by the authors; the "
        f"OSF page now displays a withdrawal tombstone and the registration "
        f"content is no longer available."
    )


def _live_cited_block(study: dict, link: str) -> str:
    """Clean control for the withdrawal abuse: a LIVE registration cited correctly.

    A real, available OSF registration referenced exactly as it should be —
    prereg_found=True with all fields extractable. The look-alike risk is only the
    surrounding 'cited' framing; nothing here is withdrawn."""
    return (
        f"This study was preregistered on the Open Science Framework; the "
        f"registration remains publicly available ({link}).\n"
        f"Hypotheses: {study['hypotheses']}\n"
        f"Design: {study['design']}\n"
        f"Sample size: {study['sample_size']}\n"
        f"Analysis plan: {study['analysis_plan']}"
    )


def _render(study: dict, form: str, rng: random.Random) -> tuple[str, dict, list[str]]:
    """Render one task's paper text and its gold + injected mistake_kinds.

    `form` is one of: osf, aspredicted, plaintext, wrong_schema_osf,
    wrong_schema_aspredicted, none, decoy, viewonly, doi_clean, embargoed,
    embargo_lifted, withdrawn, live_cited.
    Returns (paper_text, gold, mistake_kinds).
    """
    intro = rng.choice(_INTRO_OPTIONS).format(topic=study["topic"])
    method = (
        f"Participants completed the task described below; methods followed "
        f"standard procedures for research on {study['topic']}."
    )

    # --- Cases whose gold is prereg_found=False (no usable preregistration) ---
    # 'none'      : the injected no_prereg mistake (nothing filed).
    # 'decoy'     : T2 false-alarm trap (discusses prereg, embeds none) — clean.
    # 'withdrawn' : the registration was withdrawn (tombstone) yet still cited —
    #               there is no usable plan, so prereg_found=False (module 17).
    if form in ("none", "decoy", "withdrawn"):
        if form == "decoy":
            body = rng.choice(_DECOY_OPTIONS)
        elif form == "withdrawn":
            body = _withdrawn_block(study, _osf_link(study))
        else:
            body = "No preregistration was filed for this study."
        paper = "\n\n".join([intro, method, body])
        gold = {
            "prereg_found": False,
            "platform": None,
            "link": None,
            "fields": {f: None for f in FIELDS},
        }
        if form == "decoy":
            kinds: list[str] = []  # clean control
        elif form == "withdrawn":
            kinds = ["withdrawn_still_cited"]
        else:
            kinds = ["no_prereg"]
        return paper, gold, kinds

    fields = {f: study[f] for f in FIELDS}

    if form == "osf":
        link = _osf_link(study)
        block = _osf_block(study, link)
        gold = {"prereg_found": True, "platform": "osf", "link": link, "fields": fields}
        kinds: list[str] = []
    elif form == "aspredicted":
        link = _aspredicted_link(study)
        block = _aspredicted_block(study, link)
        gold = {"prereg_found": True, "platform": "aspredicted", "link": link, "fields": fields}
        kinds = []
    elif form == "plaintext":
        # Plain-text reg. Keep the platform inferable from the link domain.
        use_osf = rng.random() < 0.5
        link = _osf_link(study) if use_osf else _aspredicted_link(study)
        platform = "osf" if use_osf else "aspredicted"
        block = _plaintext_block(study, link)
        gold = {"prereg_found": True, "platform": platform, "link": link, "fields": fields}
        kinds = ["prereg_plaintext"]
    elif form == "wrong_schema_osf":
        link = _osf_link(study)
        block = _wrong_schema_block(study, link, "osf")
        gold = {"prereg_found": True, "platform": "osf", "link": link, "fields": fields}
        kinds = ["wrong_platform_schema"]
    elif form == "wrong_schema_aspredicted":
        link = _aspredicted_link(study)
        block = _wrong_schema_block(study, link, "aspredicted")
        gold = {"prereg_found": True, "platform": "aspredicted", "link": link, "fields": fields}
        kinds = ["wrong_platform_schema"]
    elif form == "viewonly":
        # Anonymized view-only OSF link instead of the canonical one. The
        # registration is real and readable: found=True, platform osf, fields
        # present; the canonical link is the plain osf.io registration URL.
        canonical = _osf_link(study)
        block = _viewonly_block(study, _osf_viewonly_link(study))
        gold = {"prereg_found": True, "platform": "osf", "link": canonical, "fields": fields}
        kinds = ["viewonly_instead_of_doi"]
    elif form == "doi_clean":
        # Clean control: canonical OSF registration DOI link. found=True,
        # platform osf, fields present.
        link = _osf_doi_link(study)
        block = _doi_block(study, link)
        gold = {"prereg_found": True, "platform": "osf", "link": link, "fields": fields}
        kinds = []
    elif form == "embargoed":
        # Registration referenced but embargoed at publication: found=True,
        # platform osf, but fields are NOT recoverable (content hidden) -> null.
        link = _osf_link(study)
        block = _embargoed_block(study, link)
        gold = {
            "prereg_found": True,
            "platform": "osf",
            "link": link,
            "fields": {f: None for f in FIELDS},
        }
        kinds = ["embargoed_at_publication"]
    elif form == "embargo_lifted":
        # Clean control: embargo lifted before publication, so fields ARE public.
        link = _osf_link(study)
        block = _embargo_lifted_block(study, link)
        gold = {"prereg_found": True, "platform": "osf", "link": link, "fields": fields}
        kinds = []
    elif form == "live_cited":
        # Clean control for withdrawn: a live registration cited correctly.
        link = _osf_link(study)
        block = _live_cited_block(study, link)
        gold = {"prereg_found": True, "platform": "osf", "link": link, "fields": fields}
        kinds = []
    else:
        raise ValueError(f"unknown form {form!r}")

    paper = "\n\n".join([intro, method, block])
    return paper, gold, kinds


def generate(task_set_version: str, seed: int, split: str = "revealed"):
    visibility = "public" if split == "revealed" else "held_out"
    studies = _load_studies()
    n = len(studies)

    def emit(tier, idx, study, form):
        tid = f"px-t{tier}-{idx}-s{seed}"
        rng = random.Random(_seed_int(task_set_version, seed, tier, idx, form))
        paper, gold_core, kinds = _render(study, form, rng)
        gold = dict(gold_core)
        gold["mistake_kinds"] = kinds
        env = {
            "task_id": tid,
            "arena_id": "prereg-extraction-v1",
            "task_set_version": "v1",
            "split": split,
            "visibility": visibility,
            "difficulty": {"tier": tier},
            "input": {"text": paper},
        }
        _GROUND_TRUTH_CACHE[tid] = gold
        return env

    # Every tier's form list is a FIXED, index-cycled sequence (no rng picks the
    # form), so the revealed and private splits produce the identical
    # tier x injected-mistake matrix → parity holds at count_tolerance 0. Only the
    # surrounding narrative WORDING is seed-driven (see _render).

    # T1: clean/simple — the clean structured controls, one per surface form, so
    # every legitimate registration form (including the three confusable
    # look-alikes for the new abuses) is represented at the floor. No mistake.
    t1_forms = ["osf", "aspredicted", "doi_clean", "embargo_lifted", "live_cited"]
    for k, form in enumerate(t1_forms):
        yield emit(1, k, studies[k % n], form)

    # T2: false-alarm trap — text that DISCUSSES preregistration but embeds no
    # registration. A good player must answer prereg_found=False. Clean controls.
    for k in range(4):
        yield emit(2, k, studies[k % n], "decoy")

    # T3: single injected mistake — cycle deterministically through ALL injected
    # kinds so every kind is covered exactly once (n studies == n injected kinds).
    t3_forms = [
        "none",            # no_prereg
        "plaintext",       # prereg_plaintext
        "wrong_schema_osf",  # wrong_platform_schema
        "viewonly",        # viewonly_instead_of_doi
        "embargoed",       # embargoed_at_publication
        "withdrawn",       # withdrawn_still_cited
    ]
    for i in range(n):
        yield emit(3, i, studies[i], t3_forms[i % len(t3_forms)])

    # T4: subtle — each new abuse placed immediately next to its confusable CLEAN
    # look-alike (viewonly↔doi_clean, embargoed↔embargo_lifted, withdrawn↔
    # live_cited), plus the unstructured plaintext reg. This adjacency is where the
    # realism lives: the player must separate the abuse from its honest twin.
    t4_forms = [
        "plaintext",
        "viewonly", "doi_clean",
        "embargoed", "embargo_lifted",
        "withdrawn", "live_cited",
    ]
    for k, form in enumerate(t4_forms):
        yield emit(4, k, studies[k % n], form)

    # T5: multiple — the schema mismatch in both directions plus the new abuses and
    # a no_prereg, recurring the full injected-mistake set in a denser tier.
    t5_forms = [
        "wrong_schema_osf", "wrong_schema_aspredicted",
        "viewonly", "embargoed", "withdrawn",
        "none", "plaintext",
    ]
    for k, form in enumerate(t5_forms):
        yield emit(5, k, studies[k % n], form)

    # T6: full composition — one task per injected kind plus every clean form,
    # spanning the entire surface-form space.
    t6_forms = [
        "none", "plaintext", "wrong_schema_osf",
        "viewonly", "embargoed", "withdrawn",
        "osf", "aspredicted", "doi_clean", "embargo_lifted", "live_cited",
    ]
    for k, form in enumerate(t6_forms):
        yield emit(6, k, studies[k % n], form)


def ground_truth(task_id: str) -> dict:
    """Return gold for a task. Regenerated from seed via the in-process cache.

    The runner always calls generate() before ground_truth(); this arena needs
    no external gold registry because the secret is the private seed, not a
    stored answer key.
    """
    if task_id not in _GROUND_TRUTH_CACHE:
        raise KeyError(
            f"No cached gold for {task_id!r}; call generate() for the matching "
            "split/seed before ground_truth()."
        )
    return _GROUND_TRUTH_CACHE[task_id]
