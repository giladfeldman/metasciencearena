"""Task provenance sidecar records: validate, write, and project for public ledgers.

The record format is `contract/schemas/task_provenance.schema.json` (field names fixed in
docs/design/2026-09-arena-expansion/task-provenance.md). A provenance record names the
injected issues, i.e. the answer key, so it never enters the player envelope and never
leaves the private split whole. `public_projection()` is the one place that decides what a
public ledger may carry for a task; do not hand-pick fields anywhere else.
"""
from __future__ import annotations

import functools
import json
from pathlib import Path

from jsonschema import Draft202012Validator

from framework.paths import schema_path


@functools.lru_cache(maxsize=1)
def _validator() -> Draft202012Validator:
    return Draft202012Validator(
        json.loads(schema_path("task_provenance.schema.json").read_text(encoding="utf-8"))
    )


def validate(record: dict) -> None:
    """Raise ValueError naming the first few violations; silent on a valid record."""
    errors = sorted(_validator().iter_errors(record), key=lambda e: list(e.path))
    if errors:
        detail = "; ".join(f"{list(e.path)}: {e.message}" for e in errors[:3])
        raise ValueError(f"invalid provenance for task {record.get('task_id')!r}: {detail}")


def write_jsonl(records: list[dict], path: Path) -> int:
    """Validate every record, then write them one per line. Writes nothing if any is invalid.

    Refuses an empty list: a provenance file with zero records for a minted task set is a
    false all-clear, not a clean result.
    """
    if not records:
        raise ValueError(f"refusing to write an empty provenance file: {path}")
    for r in records:
        validate(r)
    ids = [r["task_id"] for r in records]
    dupes = sorted({t for t in ids if ids.count(t) > 1})
    if dupes:
        raise ValueError(f"duplicate task_id in provenance: {dupes[:5]}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    return len(records)


def public_projection(record: dict) -> dict:
    """The non-leaking subset: origin, introduced_in, basis.summary, contributor display.

    ORCID appears only when the contributor chose `display: identified`. Injected issues,
    generator split/fingerprint, basis refs and review details are never included, whatever
    the task's split, so one code path serves both splits.
    """
    validate(record)
    out = {
        "task_id": record["task_id"],
        "arena_id": record["arena_id"],
        "introduced_in": record["introduced_in"],
        "origin": record["origin"],
        "basis_summary": record["basis"]["summary"],
    }
    c = record.get("contributor")
    if c is not None:
        out["contributor"] = {
            "display": c["display"],
            "pseudonym": c["pseudonym"],
            "orcid": c["orcid"] if c["display"] == "identified" else None,
            "declared_relationships": c["declared_relationships"],
        }
    return out
