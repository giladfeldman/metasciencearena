"""Shared LiteParse parsing helpers for Meta Science Arena PDF adapters."""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Iterator

from players.adapters._timeout import run_with_timeout


@dataclass(frozen=True)
class LiteparseTextItem:
    text: str
    page_num: int
    x: float
    y: float
    width: float
    height: float
    font_size: float | None


def liteparse_version() -> str | None:
    """``liteparse-<version>`` of the installed package, else None.

    Delegates to the shared detector rather than re-reading ``__version__``:
    this copy independently had the stale-attribute bug that misattributed
    liteparse run records (see ``_tool_version.module_version``).
    """
    from players.adapters._tool_version import module_version
    return module_version("liteparse")


def build_liteparse(
    *,
    ocr_enabled: bool = True,
    ocr_language: str = "eng",
    dpi: float = 150,
    quiet: bool = True,
    max_pages: int = 1000,
    preserve_very_small_text: bool = False,
    num_workers: int | None = None,
):
    """Construct a LiteParse instance (lazy-imports the package)."""
    mod = importlib.import_module("liteparse")
    kwargs: dict = {
        "ocr_enabled": ocr_enabled,
        "ocr_language": ocr_language,
        "dpi": dpi,
        "quiet": quiet,
        "max_pages": max_pages,
        "preserve_very_small_text": preserve_very_small_text,
    }
    if num_workers is not None:
        kwargs["num_workers"] = num_workers
    return mod.LiteParse(**kwargs)


def parse_pdf_bytes(
    parser,
    pdf_bytes: bytes,
    *,
    timeout_s: int,
) -> Any:
    """Parse PDF bytes with a wall-clock timeout.

    Delegates to ``run_with_timeout`` (which calls ``shutdown(wait=False)`` in a
    ``finally``) instead of a ``with ThreadPoolExecutor(...)`` block. The
    context-manager form's ``__exit__`` runs ``shutdown(wait=True)``, which would
    BLOCK on the very hung worker we are escaping — re-introducing the >10-min
    tournament hang the 2026-06-12 incident fixed for docpluck. See
    ``_timeout.py``'s module docstring for the full rationale (DR-0002).
    """
    return run_with_timeout(
        lambda: parser.parse(pdf_bytes),
        timeout_s if timeout_s > 0 else 300,
        label="liteparse parse",
    )


def pages_from_result(result: Any) -> list[str]:
    """Per-page text in document order."""
    pages = [p.text.strip() for p in result.pages if (p.text or "").strip()]
    if pages:
        return pages
    if result.text.strip():
        return [result.text.strip()]
    return [""]


def full_text_from_pages(pages: list[str]) -> str:
    return "\n\n".join(pages)


def iter_text_items(result: Any) -> Iterator[LiteparseTextItem]:
    for page in result.pages:
        for item in page.text_items:
            t = (item.text or "").strip()
            if not t:
                continue
            yield LiteparseTextItem(
                text=t,
                page_num=page.page_num,
                x=float(item.x),
                y=float(item.y),
                width=float(item.width),
                height=float(item.height),
                font_size=float(item.font_size) if item.font_size is not None else None,
            )


def strategy_suffix(
    *,
    ocr_enabled: bool,
    ocr_language: str,
    dpi: float,
    extra: str = "",
) -> str:
    parts = [f"ocr={ocr_enabled}", f"lang={ocr_language}", f"dpi={dpi}"]
    if extra:
        parts.append(extra)
    return "liteparse " + " ".join(parts)


class LiteparseOcrConfigMixin:
    """Accept the liteparse OCR/render keys from `registry.yaml` as real kwargs.

    `PlayerAdapter` is a dataclass, so a subclass that merely ANNOTATES
    `ocr_enabled: bool = True` gets a class attribute and inherits an `__init__`
    that rejects the keyword. Before 2026-08-19 `build_adapter` dropped those
    keys anyway, so the mismatch was invisible: `liteparse-no-ocr` ran WITH OCR
    and published a score bit-identical to `liteparse-default`. Once the keys
    were actually forwarded, the two heuristic adapters raised TypeError at
    construction instead.

    Mixing this in ahead of `PlayerAdapter` gives all three liteparse adapters
    one place where these kwargs are bound, so a fourth cannot drift again.
    """

    ocr_enabled: bool = True
    ocr_language: str = "eng"
    dpi: float = 150

    def __init__(
        self,
        *args,
        ocr_enabled: bool = True,
        ocr_language: str = "eng",
        dpi: float = 150,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ocr_enabled = ocr_enabled
        self.ocr_language = ocr_language
        self.dpi = dpi
