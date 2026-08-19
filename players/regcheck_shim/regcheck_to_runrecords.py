"""Map a regcheck `result.json` (general prereg-vs-paper comparison) onto the
prereg-deviation-v1 player output schema, and assemble a valid run record.

regcheck emits, per dimension, a `deviation_judgement` of "yes" / "no" /
"missing" plus free-text summaries. The arena's output schema wants, per
dimension: a boolean `deviation`, an optional `deviation_kind`, and a
`confidence` in [0,1].

Mapping rules
-------------
  deviation_judgement == "yes"      -> deviation=True,  confidence=0.9
  deviation_judgement == "no"       -> deviation=False, confidence=0.9
  deviation_judgement == "missing"  -> deviation=False, confidence=0.3  (low — unsure)
  (anything else / blank)           -> deviation=False, confidence=0.3

`deviation_kind` is NOT produced by regcheck in our label vocabulary, so when a
deviation is flagged we look up the dimension's canonical kind from the arena's
dimensions catalog (a deterministic 1:1 map). This is the honest best we can do:
regcheck tells us *whether* a dimension deviates; the arena's closed dimension
set tells us *what kind* that dimension's deviation is.

This module is import-safe (no side effects) so the adapter and a standalone
CLI run can both reuse `regcheck_items_to_output`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

ARENA_DIR = Path(__file__).resolve().parents[2] / "arenas" / "prereg-deviation-v1"
DIMENSIONS_YAML = ARENA_DIR / "catalogs" / "dimensions.yaml"

_CONF_YES = 0.9
_CONF_NO = 0.9
_CONF_MISSING = 0.3


def load_dimension_kinds() -> dict[str, str]:
    """dimension id -> canonical deviation_kind (from the arena catalog)."""
    with DIMENSIONS_YAML.open("r", encoding="utf-8") as fh:
        dims = yaml.safe_load(fh)
    return {d["id"]: d["deviation_kind"] for d in dims}


def _judgement_to_dev_conf(judgement: str) -> tuple[bool, float]:
    j = (judgement or "").strip().lower()
    if j == "yes":
        return True, _CONF_YES
    if j == "no":
        return False, _CONF_NO
    # "missing" or anything unexpected: treat as not-a-deviation, low confidence.
    return False, _CONF_MISSING


def regcheck_items_to_output(items: list[dict], dimension_kinds: dict[str, str] | None = None) -> dict:
    """Convert regcheck `result["items"]` into the arena output schema dict."""
    if dimension_kinds is None:
        dimension_kinds = load_dimension_kinds()
    deviations = []
    for item in items or []:
        dim = (item.get("dimension") or "").strip()
        if not dim:
            continue
        deviation, confidence = _judgement_to_dev_conf(item.get("deviation_judgement", ""))
        kind = dimension_kinds.get(dim) if deviation else None
        rec = {
            "dimension": dim,
            "deviation": deviation,
            "deviation_kind": kind,
            "confidence": confidence,
        }
        reg_sum = (item.get("registration_content_summary") or "").strip()
        pap_sum = (item.get("paper_content_summary") or "").strip()
        if reg_sum:
            rec["registered_summary"] = reg_sum[:500]
        if pap_sum:
            rec["paper_summary"] = pap_sum[:500]
        deviations.append(rec)
    return {"deviations": deviations}


def regcheck_result_to_output(result_json_path: str | Path) -> dict:
    data = json.loads(Path(result_json_path).read_text(encoding="utf-8"))
    items = data.get("items", []) if isinstance(data, dict) else []
    return regcheck_items_to_output(items)


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert a regcheck result.json into the prereg-deviation-v1 output schema."
    )
    parser.add_argument("result_json", help="Path to regcheck's --output JSON.")
    parser.add_argument("--out", help="Optional path to write the converted output JSON.")
    args = parser.parse_args(argv)
    output = regcheck_result_to_output(args.result_json)
    text = json.dumps(output, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
