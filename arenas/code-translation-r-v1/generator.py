"""Task generator for code-translation-r-v1.

Unlike the mistake-injection arenas, this generator does not synthesize content
from a seed: the tasks ARE the twelve curated source scripts (six analyses x two
languages) under `source_scripts/`. The seed only controls task ordering and the
split label, so `generate()` stays deterministic and parity-checkable while the
substance stays hand-authored — a translation task is only meaningful if the
source script is idiomatic, which random generation cannot deliver.

Gold is NOT stored here. It is produced by executing the hand-verified reference
R translation against the fixed dataset (`tools/build_gold.py`) and cached in
`source_scripts/gold/<analysis>.json`. Executing the gold rather than asserting
it is what makes "executable equivalence" honest.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import yaml

ARENA_DIR = Path(__file__).resolve().parent
# arenas/<id>/generator.py -> repo root. Used to store held-out dataset paths
# repo-relative in the tracked ground-truth dump (never an absolute user path).
REPO_ROOT = ARENA_DIR.parents[1]
SRC_DIR = ARENA_DIR / "source_scripts"
GOLD_DIR = SRC_DIR / "gold"
DATA_DIR = SRC_DIR / "data"

ARENA_ID = "code-translation-r-v1"

# source_language axis values. Keep this ordering stable: it is the difficulty
# axis encoding, and parity matches on it.
LANGUAGES = [("spss", 1, ".sps"), ("stata", 2, ".do")]

_GROUND_TRUTH_CACHE: dict[str, dict] = {}


def _load_yaml(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _catalog() -> list[dict]:
    return _load_yaml(SRC_DIR / "catalog.yaml")


def _datasets() -> dict:
    return _load_yaml(SRC_DIR / "datasets.yaml")


def _codebook(dataset_name: str) -> list[dict]:
    """The column dictionary handed to the player.

    Without this a translation would have to guess that 99 on item4 is a
    missing-value code — which is not a translation skill, it's a guessing game.
    The trap stays fair: the code is DECLARED, and the player must still handle it.
    """
    spec = _datasets()[dataset_name]
    out = []
    for col in spec["columns"]:
        entry = {"name": col["name"], "description": col.get("description", "")}
        if "type" in col:
            entry["type"] = col["type"]
        if "missing_code" in col:
            entry["missing_code"] = col["missing_code"]
        out.append(entry)
    return out


def load_gold(analysis_id: str) -> dict | None:
    """Executed gold statistics for one analysis, or None if not built yet."""
    p = GOLD_DIR / f"{analysis_id}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


HELD_OUT_DIR = ARENA_DIR / "task_sets" / "v1" / "_held_out"


def _held_out_cases() -> list[dict]:
    """Real third-party scripts for the private split, if any are on disk.

    Each case is a directory under ``_held_out/`` (gitignored) containing:

        <case>/meta.yaml     tier, source_language, description, gold_statistics
        <case>/source.sps    (or source.do) — the REAL script, verbatim
        <case>/data.csv      the dataset it runs against
        <case>/gold.json     executed gold, built from a verified reference

    Absent by default: this returns []. That is honest — the private split then
    re-enumerates the curated scripts under the secret seed, which is a
    seed-based holdout rather than an independent one, and the arena README says
    so. The manifest declared `real_holdout_dir` from the start but nothing read
    it until 2026-08-04; a declared-but-unread path is a promise the code does
    not keep, so this is the reader.
    """
    # Imported lazily: a generator may be loaded with only its own
    # directory on sys.path, where `framework` is not importable.
    from framework.holdout import require_corpus

    if require_corpus(HELD_OUT_DIR, arena_id='code-translation-r-v1', kind='held-out R cases') is None:
        return []
    cases = []
    for case_dir in sorted(p for p in HELD_OUT_DIR.iterdir() if p.is_dir()):
        meta_path = case_dir / "meta.yaml"
        if not meta_path.exists():
            continue
        meta = _load_yaml(meta_path)
        lang = meta.get("source_language")
        ext = {"spss": ".sps", "stata": ".do"}.get(lang)
        src = case_dir / f"source{ext}" if ext else None
        gold_path = case_dir / "gold.json"
        data_path = case_dir / "data.csv"
        # Skip loudly-incomplete cases rather than emitting a task that cannot
        # be scored — a half-populated holdout would silently score every player 0.
        if not (src and src.exists() and gold_path.exists() and data_path.exists()):
            continue
        meta["_dir"] = case_dir
        meta["_source_code"] = src.read_text(encoding="utf-8")
        meta["_gold"] = json.loads(gold_path.read_text(encoding="utf-8"))
        cases.append(meta)
    return cases


def generate(task_set_version: str, seed: int, split: str = "revealed") -> Iterable[dict]:
    """Yield one task envelope per (analysis x language) pair.

    The REVEALED split is always the curated matrix. The PRIVATE split prefers
    real third-party scripts from ``_held_out/`` when present, and otherwise
    falls back to the curated matrix under the secret seed.
    """
    visibility = "public" if split == "revealed" else "held_out"
    datasets = _datasets()

    if split != "revealed":
        real = _held_out_cases()
        if real:
            yield from _generate_held_out(task_set_version, seed, real)
            return

    for entry in _catalog():
        analysis_id = entry["id"]
        tier = entry["tier"]
        dataset_name = entry["dataset"]
        required = entry["gold_statistics"]

        for lang, lang_axis, ext in LANGUAGES:
            src = SRC_DIR / lang / f"{analysis_id}{ext}"
            if not src.exists():
                raise FileNotFoundError(f"missing source script: {src}")

            task_id = f"xlat-{analysis_id}-{lang}-s{seed}"
            envelope = {
                "task_id": task_id,
                "arena_id": ARENA_ID,
                "task_set_version": task_set_version,
                "split": split,
                "visibility": visibility,
                "difficulty": {"tier": tier, "source_language": lang_axis},
                "input": {
                    "source_language": lang,
                    "source_code": src.read_text(encoding="utf-8"),
                    "dataset_path": f"{dataset_name}.csv",
                    "codebook": _codebook(dataset_name),
                    "required_statistics": list(required),
                    # Carries any convention the statistic names alone leave
                    # ambiguous (e.g. that mean_diff is group1 - group2). The
                    # arena tests cross-language DEFAULTS, not the player's
                    # ability to guess our sign conventions.
                    "analysis_description": entry.get("description", "").strip(),
                },
            }

            _GROUND_TRUTH_CACHE[task_id] = {
                "analysis_id": analysis_id,
                "source_language": lang,
                "tier": tier,
                "dataset": dataset_name,
                "required_statistics": list(required),
                # `mistake_kinds` is the parity vocabulary every arena exposes.
                # Here the "kinds" are the cross-language traps this task carries,
                # so check_parity can confirm both splits exercise the same ones.
                "mistake_kinds": _trap_kinds(entry),
                "gold_statistics": load_gold(analysis_id),
            }
            yield envelope


def _generate_held_out(task_set_version: str, seed: int,
                       cases: list[dict]) -> Iterable[dict]:
    """Emit envelopes for REAL third-party scripts (the independent holdout).

    Each case carries its own dataset, so `dataset_path` names a file inside the
    case directory rather than the shared fixture. Gold is cached exactly as for
    the curated tasks, and `redact_ground_truth_entry` strips it from the tracked
    dump because these envelopes are marked held_out.
    """
    for case in cases:
        case_id = case["_dir"].name
        lang = case["source_language"]
        lang_axis = 1 if lang == "spss" else 2
        task_id = f"xlat-holdout-{case_id}-{lang}-s{seed}"
        required = list(case.get("gold_statistics") or case["_gold"].keys())

        yield {
            "task_id": task_id,
            "arena_id": ARENA_ID,
            "task_set_version": task_set_version,
            "split": "private",
            "visibility": "held_out",
            "difficulty": {"tier": int(case["tier"]), "source_language": lang_axis},
            "input": {
                "source_language": lang,
                "source_code": case["_source_code"],
                "dataset_path": "data.csv",
                "codebook": case.get("codebook") or [],
                "required_statistics": required,
                "analysis_description": (case.get("description") or "").strip(),
            },
        }

        _GROUND_TRUTH_CACHE[task_id] = {
            "analysis_id": f"holdout-{case_id}",
            "source_language": lang,
            "tier": int(case["tier"]),
            # Real cases carry their OWN data file; dataset_csv() resolves it.
            # Stored REPO-RELATIVE with forward slashes: this value lands in the
            # tracked _ground_truth.json dump, and an absolute path there both
            # breaks on every other machine and bakes one developer's home
            # directory into a published artifact.
            "dataset": (case["_dir"] / "data.csv").relative_to(REPO_ROOT).as_posix(),
            "required_statistics": required,
            "mistake_kinds": list(case.get("mistake_kinds") or []),
            "gold_statistics": case["_gold"],
        }


def _trap_kinds(entry: dict) -> list[str]:
    """Stable trap labels per analysis, for parity matching.

    Derived from the tier rather than parsed out of the prose in catalog.yaml,
    so the label set cannot drift when someone edits a description.
    """
    return {
        1: [],
        2: ["pooled_vs_welch"],
        3: ["listwise_deletion"],
        4: ["user_missing_code", "min_valid_count"],
        5: ["ss_type_iii", "sum_to_zero_contrasts"],
        6: ["filter_order"],
        # Tiers 7-9 (2026-08-04): added because T1-T6 saturated — sonnet-5 scored
        # 1.000 on every task, so the arena had stopped discriminating.
        7: ["pairwise_vs_listwise"],
        8: ["frequency_weight_state"],
        9: ["split_file_state"],
    }[entry["tier"]]


def ground_truth(task_id: str) -> dict:
    """Gold for one task. `generate()` must have run first (cache warm).

    The cold-cache fallback re-derives the seed from the task_id's ``-s<seed>``
    suffix rather than assuming 0: private-split ids (and every held-out id)
    carry the secret seed, so a hardcoded 0 regenerated the wrong split and the
    lookup raised KeyError for tasks that exist.
    """
    if task_id not in _GROUND_TRUTH_CACHE:
        seed = 0
        tail = task_id.rsplit("-s", 1)
        if len(tail) == 2 and tail[1].isdigit():
            seed = int(tail[1])
        for split in ("revealed", "private"):
            for _ in generate("v1", seed, split):
                pass
            if task_id in _GROUND_TRUTH_CACHE:
                break
    if task_id not in _GROUND_TRUTH_CACHE:
        raise KeyError(f"unknown task_id: {task_id}")
    return _GROUND_TRUTH_CACHE[task_id]


def dataset_csv(dataset_name: str) -> Path:
    """Resolve a ground_truth `dataset` field to a CSV on disk.

    Curated tasks store a NAME (`"wellbeing"`) resolved against the shared
    fixture dir. Real held-out cases store a REPO-RELATIVE path to the case's own
    data.csv (e.g. ``arenas/.../_held_out/<case>/data.csv``), because each real
    script comes with its own dataset. Accepting both keeps the scorer's single
    call site unchanged.

    Absolute paths are still honoured so that ground-truth dumps written before
    2026-08-06 — when this field held one developer's home directory — keep
    resolving on the machine that produced them.
    """
    p = Path(dataset_name)
    if p.is_absolute():
        return p
    if p.suffix == ".csv":
        # Repo-relative (contains a separator) vs a bare "foo.csv" beside the
        # shared fixtures.
        return REPO_ROOT / p if len(p.parts) > 1 else DATA_DIR / p
    return DATA_DIR / f"{dataset_name}.csv"
