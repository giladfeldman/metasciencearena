"""Spatial table clustering from LiteParse text_items (conservative)."""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

from players.adapters._liteparse_common import LiteparseTextItem

_CAPTION_RE = re.compile(r"^\s*(?:table|tab\.?)\s*(\d+|[ivxlc]+)\b", re.I)


@dataclass
class _Cell:
    r: int
    c: int
    text: str
    is_header: bool


def _cluster_rows(items: list[LiteparseTextItem], y_tol: float) -> list[list[LiteparseTextItem]]:
    """Group items into rows by y-center."""
    if not items:
        return []
    sorted_items = sorted(items, key=lambda it: (it.y + it.height / 2, it.x))
    rows: list[list[LiteparseTextItem]] = []
    current: list[LiteparseTextItem] = [sorted_items[0]]
    cy = sorted_items[0].y + sorted_items[0].height / 2
    for it in sorted_items[1:]:
        iy = it.y + it.height / 2
        if abs(iy - cy) <= y_tol:
            current.append(it)
        else:
            rows.append(sorted(current, key=lambda x: x.x))
            current = [it]
            cy = iy
    rows.append(sorted(current, key=lambda x: x.x))
    return rows


def _rows_to_grid(rows: list[list[LiteparseTextItem]]) -> tuple[int, int, list[_Cell]] | None:
    if len(rows) < 2:
        return None
    col_counts = [len(r) for r in rows]
    if max(col_counts) < 2:
        return None
    # Require at least 2 rows with >=2 cells
    multi = sum(1 for c in col_counts if c >= 2)
    if multi < 2:
        return None
    n_cols = max(col_counts)
    if n_cols > 20 or len(rows) > 80:
        return None
    cells: list[_Cell] = []
    for ri, row in enumerate(rows):
        for ci, it in enumerate(row):
            cells.append(_Cell(r=ri, c=ci, text=it.text, is_header=(ri == 0)))
    return len(rows), n_cols, cells


def detect_tables_from_items(
    items: Sequence[LiteparseTextItem],
    *,
    min_items: int = 6,
    y_tol_factor: float = 0.6,
) -> list[dict]:
    """Conservative table detection; returns arena-shaped table dicts."""
    by_page: dict[int, list[LiteparseTextItem]] = defaultdict(list)
    for it in items:
        by_page[it.page_num].append(it)

    tables: list[dict] = []
    table_idx = 0
    for page_num in sorted(by_page):
        page_items = by_page[page_num]
        if len(page_items) < min_items:
            continue
        heights = [it.height for it in page_items if it.height > 0]
        median_h = sorted(heights)[len(heights) // 2] if heights else 10.0
        y_tol = max(median_h * y_tol_factor, 4.0)
        rows = _cluster_rows(page_items, y_tol)
        grid = _rows_to_grid(rows)
        if grid is None:
            continue
        n_rows, n_cols, cells = grid
        # Confidence: need rectangular-ish grid (most rows same width within 1)
        widths = [len(r) for r in rows]
        if max(widths) - min(widths) > 2:
            continue
        caption = None
        top_y = min(it.y for it in page_items)
        for it in page_items:
            if it.y < top_y + median_h * 2 and _CAPTION_RE.match(it.text):
                caption = it.text
                break
        cells_out = [
            {
                "r": c.r,
                "c": c.c,
                "rowspan": 1,
                "colspan": 1,
                "text": c.text,
                "is_header": c.is_header,
            }
            for c in cells
        ]
        table_idx += 1
        tables.append({
            "label": f"table_{table_idx}",
            "page": page_num,
            "caption": caption,
            "n_rows": n_rows,
            "n_cols": n_cols,
            "header_rows": 1,
            "cells": cells_out,
        })
    return tables


def detect_tables_from_text_pages(
    pages: list[tuple[int, str]],
) -> list[dict]:
    """Fallback when no bboxes: no tables (conservative)."""
    return []
