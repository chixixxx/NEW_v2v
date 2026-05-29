from __future__ import annotations

from collections.abc import Iterable
from typing import TypeVar

T = TypeVar("T")

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional at runtime.
    tqdm = None


def progress(iterable: Iterable[T], *, desc: str, total: int | None = None, unit: str = "it") -> Iterable[T]:
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, total=total, unit=unit)

