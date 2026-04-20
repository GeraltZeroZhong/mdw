from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ProgressEvent:
    current: int
    total: int
    stage: str
    detail: str
    subcurrent: int = 0
    subtotal: int = 0
    subdetail: str = ""
    subeta_seconds: int | None = None


ProgressCallback = Callable[[ProgressEvent], None]


def _normalize_progress(current: int, total: int) -> tuple[int, int]:
    safe_total = max(int(total), 1)
    safe_current = min(max(int(current), 0), safe_total)
    return safe_current, safe_total


def emit_progress(
    callback: ProgressCallback | None,
    current: int,
    total: int,
    stage: str,
    detail: str = "",
    *,
    subcurrent: int = 0,
    subtotal: int = 0,
    subdetail: str = "",
    subeta_seconds: int | None = None,
) -> None:
    safe_current, safe_total = _normalize_progress(current, total)
    safe_subcurrent = 0
    safe_subtotal = 0
    if int(subtotal) > 0:
        safe_subcurrent, safe_subtotal = _normalize_progress(subcurrent, subtotal)
    event = ProgressEvent(
        current=safe_current,
        total=safe_total,
        stage=stage,
        detail=detail or stage.replace("_", " ").strip(),
        subcurrent=safe_subcurrent,
        subtotal=safe_subtotal,
        subdetail=subdetail or "",
        subeta_seconds=None if subeta_seconds is None else max(int(subeta_seconds), 0),
    )
    if callback is None:
        return
    callback(event)
