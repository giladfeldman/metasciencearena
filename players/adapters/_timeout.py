"""Wall-clock timeout for in-process adapters.

The runner passes a per-task ``timeout_s`` but in-process library adapters
(docpluck) historically ignored it and called the library synchronously — so a
single hard real PDF could hang the whole tournament indefinitely (observed
2026-06-12: a held-out PMC PDF stalled docpluck for >10 min with no record
written). Wrapping the work here turns a hang into a soft ``TimeoutError`` that
the runner records as an adapter error (fail-soft; Finding 7 of the accuracy
handoff) and moves on.

We deliberately do NOT use ``with ThreadPoolExecutor(...)``: its ``__exit__``
calls ``shutdown(wait=True)``, which would block on the very hung thread we are
trying to escape. ``shutdown(wait=False)`` lets the caller continue; the orphan
worker thread is harmless (it dies with the process) and the next task proceeds.
A pure-Python infinite loop that never releases the GIL can still defeat this,
but real PDF hangs live in C extensions / subprocesses that release it.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Callable, TypeVar

T = TypeVar("T")


def run_with_timeout(fn: Callable[[], T], timeout_s: int | None, *, label: str = "task") -> T:
    """Run ``fn()`` with a wall-clock timeout. Raises ``TimeoutError`` on expiry."""
    if timeout_s is None or timeout_s <= 0:
        return fn()
    pool = ThreadPoolExecutor(max_workers=1)
    fut = pool.submit(fn)
    try:
        return fut.result(timeout=timeout_s)
    except FuturesTimeoutError as exc:
        raise TimeoutError(f"{label} exceeded {timeout_s}s") from exc
    finally:
        pool.shutdown(wait=False)
