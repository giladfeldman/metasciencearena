"""docpluck >= 2.4.145 `caption_status="uncaptioned_candidate"` tables are EMITTED and counted.

docpluck asked (thread M-docpluck-arena-20260927T102719Z, item A1) that these be treated
as "not a claimed table". The owner decided on 2026-09-28 to score them like any other
table instead: this player is maintainer-built, and withholding them favoured it
(held-out .6480 withheld vs .5432 emitted at 2.4.146). These tests pin that decision so
a later "fix" cannot quietly flip it back, and they pin the count that makes the
candidates visible in the record.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from players.adapters.docpluck_tables import DocpluckTablesAdapter

REPO = Path(__file__).resolve().parents[3]


def _scorer():
    spec = importlib.util.spec_from_file_location(
        "tables_scorer", REPO / "arenas" / "pdf-table-extraction-v1" / "scorer.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _table(tid, label, caption, status):
    return {"id": tid, "label": label, "caption": caption, "page": 2, "kind": "structured",
            "n_rows": 1, "n_cols": 1, "header_rows": 0, "caption_status": status,
            "cells": [{"r": 0, "c": 0, "rowspan": 1, "colspan": 1, "text": "x", "is_header": False}]}


class _FakeDocpluck:
    def __init__(self, tables):
        self._tables = tables

    def extract_pdf_structured(self, pdf_bytes, **kw):
        return {"tables": self._tables, "table_extraction_version": "2.4.17"}


def _run(tables):
    a = DocpluckTablesAdapter.__new__(DocpluckTablesAdapter)
    a._dp = _FakeDocpluck(tables)
    return a._extract(b"%PDF")


GOLD = {"gold_tables": [{"label": "Table 1", "caption": "Means", "page": 2, "n_rows": 1,
                         "n_cols": 1, "header_rows": 0,
                         "cells": [{"r": 0, "c": 0, "rowspan": 1, "colspan": 1, "text": "x", "is_header": False}]}]}


def test_uncaptioned_candidate_is_emitted_and_counted():
    out = _run([_table("t1", "Table 1", "Table 1. Means", "matched"),
                _table("u1", None, None, "uncaptioned_candidate")])
    assert [t["label"] for t in out["tables"]] == ["Table 1", None]
    assert "uncaptioned_candidates_emitted=1" in out["player_strategy_notes"]


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[3] / "arenas" / "pdf-table-extraction-v1" / "scorer.py").is_file(),
    reason="pdf-table-extraction-v1 is a PRIVATE arena; its scorer is not in the public mirror",
)
def test_an_emitted_candidate_is_charged_by_the_scorer():
    """Control: the candidate costs a table_extra against all-labelled gold, so the
    emit rule is a real scoring decision, not a no-op."""
    sc = _scorer()
    with_candidate = _run([_table("t1", "Table 1", "Table 1. Means", "matched"),
                           _table("u1", None, None, "uncaptioned_candidate")])
    without = _run([_table("t1", "Table 1", "Table 1. Means", "matched")])
    s_with, s_without = sc.score(with_candidate, GOLD), sc.score(without, GOLD)
    assert s_with["primary"] < s_without["primary"]
    assert any(f["category"] == "table_extra" for f in s_with["findings"])


def test_other_statuses_and_pre_2_4_145_output_pass_through():
    out = _run([_table("t1", "Table 1", "Table 1. Means", "matched"),
                _table("t2", "Table 2", None, "none_found"),
                {k: v for k, v in _table("t3", "Table 3", "Table 3. SDs", None).items()
                 if k != "caption_status"}])
    assert [t["label"] for t in out["tables"]] == ["Table 1", "Table 2", "Table 3"]
    assert "uncaptioned_candidates_emitted=0" in out["player_strategy_notes"]
