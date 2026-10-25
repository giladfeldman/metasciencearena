"""The PDF universe: one shared corpus of synthetic papers feeding every PDF arena.

`task_specs(version)` lists every task (deterministic). `built(spec)` returns the
rendered PDF + gold for one spec, cached on disk under tmp/ (gitignored) and keyed by
the spec AND a hash of the pdfforge source, so a code change rebuilds instead of
serving a stale PDF. Every arena's generator maps the same specs to its own envelope
and gold, so improving the universe improves all of them at once.

Splits:
  revealed (visibility public)  -- seeds derived from the public base seed
  private  (visibility held_out) -- seeds derived from the universe's private seed,
           kept outside git (arenas_private/pdf-universe/.private_seed, backed up in
           ~/sciencearena_private_seeds.backup.txt). Without it, the private split is
           ASSET-ABSENT (reported, never silently skipped as a pass).

Composition per version (v1):
  isolation  each admissible recipe alone, in 2 papers (one revealed, one private)
  twins      the clean twin of the revealed isolation paper (catalog X01)
  house      12 publisher-like profiles x 10 papers (4 revealed, 6 private)
  controls   image-only pages (X03: text layer absent)
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PKG = Path(__file__).resolve().parent
PROOF = PKG / "recipe_proof.json"
BANK = PKG / "prose_bank.json"
PRIVATE_SEED_FILE = REPO / "arenas_private" / "pdf-universe" / ".private_seed"
# Outside the project (d-ff1794): the pickles hold whole synthetic papers, and inside the
# repo -- even gitignored -- the portfolio custody scan counts each as an article-text leak.
CACHE = Path(os.environ.get("SCIENCEARENA_PDFFORGE_CACHE")
             or Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".cache") / "sciencearena" / "pdfforge_cache")
PUBLIC_BASE_SEED = 20261017
#: Seed offsets chosen once by tools/select_seeds.py so every isolation recipe finds a
#: statistic it can act on and shows up in an extractor (committed: deterministic corpus).
OFFSETS = PKG / "universe_seed_offsets.json"


class PrivateSeedMissing(RuntimeError):
    pass


@dataclass(frozen=True)
class Spec:
    task_key: str              # stable, opaque key (no recipe name leaks for private tasks)
    kind: str                  # isolation | twin | house | control
    split: str                 # revealed | private
    seed: int
    topic: str
    defects: tuple = ()
    layout: tuple = ()         # tuple of (key, value)
    style_pair: tuple | None = None
    house: str | None = None
    n_studies: int | None = None
    note: str = ""

    @property
    def visibility(self) -> str:
        return "public" if self.split == "revealed" else "held_out"


HOUSE_STYLES: dict[str, dict] = {
    "elsevier_2col": {"style": ("apa", "apa"), "layout": {"columns": 2}, "defects": ["P01", "P04", "P05", "F18", "T03", "B01"]},
    "sage_quarter": {"style": ("apa", "apa"), "layout": {"columns": 1}, "defects": ["F08", "P05", "P01", "S05"]},
    "collabra_stamp": {"style": ("apa", "apa"), "layout": {"columns": 1}, "defects": ["P03", "F13", "F02", "T01"]},
    "cambridge_jdm": {"style": ("apa", "apa"), "layout": {"columns": 1}, "defects": ["P02", "P03", "F01", "F11", "T16"]},
    "nature_numeric": {"style": ("superscript", "nature"), "layout": {"columns": 2}, "defects": ["T06", "C02", "C03", "F02"]},
    "pnas_numeric": {"style": ("numeric", "vancouver"), "layout": {"columns": 2}, "defects": ["C04", "H01", "B07", "L01"]},
    "apa_journal": {"style": ("apa", "apa"), "layout": {"columns": 1}, "defects": ["P01", "H02", "S11", "R05a"]},
    "pmc_manuscript": {"style": ("numeric", "vancouver"), "layout": {"columns": 1}, "defects": ["P09", "F21", "B08"]},
    "preprint_lines": {"style": ("apa", "apa"), "layout": {"columns": 1}, "defects": ["P10", "L23", "S12", "P08"]},
    "rsos_2col": {"style": ("harvard", "harvard"), "layout": {"columns": 2}, "defects": ["T17", "F29", "F03", "B04"]},
    "economics_aom": {"style": ("aom", "aom"), "layout": {"columns": 1}, "defects": ["C01", "T09", "B04", "C07"]},
    "ieee_tech": {"style": ("numeric", "ieee"), "layout": {"columns": 2}, "defects": ["B12", "H01", "T06", "F06"]},
}

# Recipes whose isolation task needs a particular citation style / study count.
_NEEDS_STYLE = {"C02": ("superscript", "nature"), "C03": ("superscript", "nature"), "C04": ("numeric", "vancouver"),
                "B07": ("numeric", "vancouver"), "B08": ("numeric", "vancouver"), "B12": ("numeric", "ieee")}
_AUTHOR_YEAR_RECIPES = {"B06", "C01", "C05", "C06", "C07", "C08", "C09", "C10", "C13", "C14"}


def private_seed(required: bool = True) -> int | None:
    env = os.environ.get("SCIENCEARENA_PDFU_PRIVATE_SEED")
    if env:
        return int(env)
    if PRIVATE_SEED_FILE.is_file():
        return int(PRIVATE_SEED_FILE.read_text().strip())
    if required:
        raise PrivateSeedMissing(f"PDF-universe private seed not found at {PRIVATE_SEED_FILE}")
    return None


def _admissible() -> list[str]:
    proof = json.loads(PROOF.read_text(encoding="utf-8"))
    return sorted(k for k, v in proof.items() if v.get("applied") and v.get("manifests") and not v.get("error"))


def _seed(base: int, *parts) -> int:
    h = hashlib.sha256("|".join(map(str, (base,) + parts)).encode()).hexdigest()
    return int(h[:12], 16)


def _opaque(split: str, *parts) -> str:
    return "pu1-" + hashlib.sha256("|".join(map(str, (split,) + parts)).encode()).hexdigest()[:10]


def task_specs(version: str = "v1", *, include_private: bool = True) -> list[Spec]:
    if version != "v1":
        raise ValueError(f"unknown PDF-universe version {version!r}")
    bank = json.loads(BANK.read_text(encoding="utf-8"))["entries"]
    topics = sorted(bank)
    # topics are locked to a split, so no prose ever appears in both
    rev_topics = [t for t in topics if int(hashlib.sha256(t.encode()).hexdigest(), 16) % 3 == 0]
    prv_topics = [t for t in topics if t not in rev_topics]
    pseed = private_seed(required=include_private) if include_private else None
    specs: list[Spec] = []

    def topic(split, k):
        pool = rev_topics if split == "revealed" else prv_topics
        return pool[k % len(pool)]

    offsets = json.loads(OFFSETS.read_text(encoding="utf-8")) if OFFSETS.exists() else {}
    for i, did in enumerate(_admissible()):
        for split in ("revealed", "private"):
            if split == "private" and pseed is None:
                continue
            base = PUBLIC_BASE_SEED if split == "revealed" else pseed
            off = offsets.get(f"iso|{split}|{did}", 0)
            seed = _seed(base, "iso", did) if off == 0 else _seed(base, "iso", did, off)
            style = _NEEDS_STYLE.get(did)
            if did in _AUTHOR_YEAR_RECIPES:
                style = ("apa", "apa")
            n_st = 2 if did == "H05" else None
            key = (f"pu1-iso-{did}" if split == "revealed" else _opaque(split, "iso", did, pseed))
            lay = (("columns", 2 if i % 2 else 1),)
            specs.append(Spec(key, "isolation", split, seed, topic(split, i), (did,), lay, style, None, n_st))
            if split == "revealed":
                specs.append(Spec(f"pu1-twin-{did}", "twin", split, seed, topic(split, i), (), lay, style, None, n_st,
                                  note=f"clean twin of pu1-iso-{did}"))
    extras = _admissible()
    for h, (name, prof) in enumerate(sorted(HOUSE_STYLES.items())):
        for k in range(10):
            split = "revealed" if k < 4 else "private"
            if split == "private" and pseed is None:
                continue
            base = PUBLIC_BASE_SEED if split == "revealed" else pseed
            seed = _seed(base, "house", name, k)
            rng = random.Random(seed)
            add = [d for d in rng.sample(extras, 3) if d[0] in "SRTL" and d not in prof["defects"]]
            key = f"pu1-house-{name}-{k}" if split == "revealed" else _opaque(split, "house", name, k, pseed)
            specs.append(Spec(key, "house", split, seed, topic(split, h * 10 + k), tuple(prof["defects"] + add),
                              tuple(sorted(prof["layout"].items())), prof["style"], name))
    for k in range(4):
        split = "revealed" if k < 2 else "private"
        if split == "private" and pseed is None:
            continue
        base = PUBLIC_BASE_SEED if split == "revealed" else pseed
        seed = _seed(base, "ctrl", k)
        key = f"pu1-ctrl-image-{k}" if split == "revealed" else _opaque(split, "ctrl", k, pseed)
        specs.append(Spec(key, "control", split, seed, topic(split, 900 + k), (), (("columns", 1),), None, None,
                          note="X03: pages rendered as images, no text layer"))
    return specs


def _source_hash() -> str:
    h = hashlib.sha256()
    for p in sorted(PKG.rglob("*.py")):
        if "tests" in p.parts or "tools" in p.parts:
            continue
        h.update(p.read_bytes())
    for p in sorted((PKG / "fonts").glob("*.ttf")):
        h.update(p.name.encode())
        h.update(str(p.stat().st_size).encode())
    h.update(BANK.read_bytes())
    return h.hexdigest()[:16]


_SRC = None


def built(spec: Spec):
    """Build (or load from cache) one task. Returns build.Built."""
    global _SRC
    from .build import build_task
    if _SRC is None:
        _SRC = _source_hash()
    key = hashlib.sha256(json.dumps(asdict(spec), sort_keys=True).encode() + _SRC.encode()).hexdigest()[:24]
    path = CACHE / f"{key}.pkl"
    if path.exists():
        with path.open("rb") as f:
            return pickle.load(f)
    bank = json.loads(BANK.read_text(encoding="utf-8"))["entries"]
    kw = {}
    if spec.style_pair:
        kw["style_pair"] = tuple(spec.style_pair)
    if spec.n_studies:
        kw["n_studies"] = spec.n_studies
    bt = build_task(spec.task_key, spec.seed, bank[spec.topic], spec.split, list(spec.defects),
                    dict(spec.layout), proof=spec.kind in ("isolation", "house"), **kw)
    if spec.kind == "control":
        from .controls import image_only
        bt.pdf = image_only(bt.pdf)
        bt.proof["control"] = "image_only"
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        pickle.dump(bt, f)
    tmp.replace(path)
    return bt
