"""Generator for citation-claim-fidelity-v1.

Revealed split: the fictional examples in catalogs/revealed_examples.yaml.
Private split: the real task set built by tools/build_tasks.py, read from article
custody ($ARTICLE_REPOSITORY/ground_truth/citeclaim-2026-10/build_vN/tasks_vN.jsonl)
and checked against the sha256 recorded in that folder's MANIFEST.json. No real
article text is ever stored in this repository. A missing or altered custody file
RAISES; it never yields an empty or partial task set.

Held-out task ids AND the held-out gold are salted with the private seed, so a
tracked run record names no DOI, and neither its task id nor its gold_sha256 can
be recomputed from a guessed paper.

Every task: one abstract and k citing sentences, rendered into the arena-owned
prompt (prompts/v1.md) through input.prompt_fields.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import yaml

ARENA_DIR = Path(__file__).resolve().parent
KNOWN_TASK_SET_VERSIONS = ("v1", "v2", "v3", "v4")
CUSTODY_REL = ("ground_truth", "citeclaim-2026-10")
#: v2 = every v1 task unchanged + new tasks from seeds that state a number their
#: abstract states (C1 top-up; owner decision 2026-10-15). tools/build_tasks.py.
#: v3 = every v2 task unchanged (tier "synthetic") + the real-data tiers of citeclaim Stage D:
#: human-labelled public datasets, one sentence per task (tools/build_real_tier.py).
#: v4 = every v3 task unchanged + tier rd-mmc-pilot of citeclaim Stage E: real MetaMisCitations
#: citing sentences grouped per source as the library groups them, FRONTIER labels (three-seat
#: consensus or a blind Opus decider; tools/build_mmc_tier.py).
TASKS_REL_BY_VERSION = {"v1": "build_v1/tasks_v1.jsonl", "v2": "build_v2/tasks_v2.jsonl",
                        "v3": "build_v3/tasks_v3.jsonl", "v4": "build_v4/tasks_v4.jsonl"}
#: v3/v4 tiers. A task without a "tier" key is synthetic (every v1/v2 task).
TIERS = ("synthetic", "rd-zhang2024", "rd-scifact-dev", "rd-mmc-pilot")
TASKS_REL = TASKS_REL_BY_VERSION["v1"]

_GROUND_TRUTH_CACHE: dict[str, dict] = {}


class CustodyTasksUnavailable(RuntimeError):
    """The private task set could not be read from article custody, or failed its hash."""


def render_sentences(sentences: list[str]) -> str:
    """Identical to the citeclaim library's renderSentences."""
    return "\n".join(f'{i + 1}. "{s}"' for i, s in enumerate(sentences))


def _envelope(task_id: str, task: dict, split: str, visibility: str, salt: str | None = None,
              version: str = "v1") -> tuple[dict, dict]:
    items = task["items"]
    sentences = [it["sentence"] for it in items]
    envelope = {
        "task_id": task_id,
        "arena_id": "citation-claim-fidelity-v1",
        "task_set_version": version,
        "split": split,
        "visibility": visibility,
        "difficulty": {"group_size": len(items),
                       "n_not_consistent": sum(1 for it in items if it["gold"] != "consistent")},
        "input": {
            "group_size": len(items),
            "prompt_fields": {"title": task.get("title") or "(unknown)",
                              "year": task.get("year") if task.get("year") not in (None, "") else "n.d.",
                              "abstract": task["abstract"], "sentences": render_sentences(sentences)},
        },
    }
    gold = {"abstract": task["abstract"],
            "items": [{"id": i + 1, "gold": it["gold"], "type": it["type"], "traps": it.get("traps", [])}
                      for i, it in enumerate(items)]}
    if version not in ("v1", "v2"):
        # v3+: the tier and the dataset's own per-item metadata (source, explanation) travel
        # with the gold, for the per-tier bars and the by-source report. Never added to v1/v2
        # gold, whose hashes are recorded on every past run record.
        gold["tier"] = task.get("tier", "synthetic")
        for g, it in zip(gold["items"], items):
            if it.get("meta"):
                g["meta"] = it["meta"]
    if salt is not None:
        # A tracked held-out record carries instrument.gold_sha256. Unsalted, that
        # hash would let anyone holding a candidate paper's public abstract confirm
        # its membership (Sol, consult 2026-10-14). The salt is derived from the
        # private seed, which never leaves this machine.
        gold["salt"] = salt
    return envelope, gold


def _revealed_tasks() -> list[dict]:
    cat = yaml.safe_load((ARENA_DIR / "catalogs" / "revealed_examples.yaml").read_text(encoding="utf-8"))
    return [{"key": t["key"], "title": t["title"], "year": t["year"], "abstract": " ".join(t["abstract"].split()),
             "items": t["items"]} for t in cat["tasks"]]


def custody_dir() -> Path:
    repo = os.environ.get("ARTICLE_REPOSITORY")
    if not repo:
        raise CustodyTasksUnavailable("ARTICLE_REPOSITORY is not set; the private split lives in article custody")
    return Path(repo).joinpath(*CUSTODY_REL)


def _private_tasks(version: str = "v1") -> list[dict]:
    base = custody_dir()
    rel = TASKS_REL_BY_VERSION[version]
    path = base / rel
    manifest = base / "MANIFEST.json"
    if not path.exists() or not manifest.exists():
        raise CustodyTasksUnavailable(f"{rel} or MANIFEST.json missing under {base}")
    expected = (json.loads(manifest.read_text(encoding="utf-8")).get(rel) or {}).get("sha256")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if not expected or expected != actual:
        raise CustodyTasksUnavailable(f"{rel} sha256 {actual[:12]} does not match MANIFEST {str(expected)[:12]}")
    tasks = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not tasks:
        raise CustodyTasksUnavailable(f"{rel} is empty")
    return tasks


def generate(task_set_version: str, seed: int, split: str = "revealed"):
    if task_set_version not in KNOWN_TASK_SET_VERSIONS:
        raise ValueError(f"citation-claim-fidelity-v1 has no task set {task_set_version!r}")
    if split == "revealed":
        for t in _revealed_tasks():
            tid = f"ccf-r-{t['key']}"
            env, gt = _envelope(tid, t, "revealed", "public", version=task_set_version)
            _GROUND_TRUTH_CACHE[tid] = gt
            yield env
        return
    for t in _private_tasks(task_set_version):
        salted = hashlib.sha256(f"{seed}|{t['task_key']}".encode("utf-8")).hexdigest()[:16]
        tid = f"ccf-p-{salted}"
        salt = hashlib.sha256(f"gold-salt|{seed}|{t['task_key']}".encode("utf-8")).hexdigest()
        env, gt = _envelope(tid, t, "private", "held_out", salt=salt, version=task_set_version)
        _GROUND_TRUTH_CACHE[tid] = gt
        yield env


def ground_truth(task_id: str) -> dict:
    if task_id not in _GROUND_TRUTH_CACHE:
        raise KeyError(f"No cached gold for {task_id!r}; call generate() for the matching split/seed first.")
    return _GROUND_TRUTH_CACHE[task_id]
