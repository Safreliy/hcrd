"""Dataset-specific parsing kept separate from the signal transform."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np


def select_cwru_drive_key(record_id: int, keys: Iterable[str]) -> str:
    """Select the drive-end variable matching the MAT filename record id.

    Official record 99 also contains copied variables from record 98, so merely
    requiring a unique ``*_DE_time`` suffix is not sufficient.
    """

    expected = f"X{record_id:03d}_DE_time"
    available = list(keys)
    if expected not in available:
        candidates = [key for key in available if key.lower().endswith("_de_time")]
        raise RuntimeError(
            f"expected CWRU variable {expected!r}; drive-end candidates are {candidates}"
        )
    return expected


def load_cwru_drive_end(path: Path) -> tuple[np.ndarray, float | None]:
    from scipy.io import loadmat

    try:
        record_id = int(path.stem)
    except ValueError as error:
        raise ValueError(f"CWRU MAT filename must be a numeric record id: {path}") from error
    content = loadmat(path)
    signal_key = select_cwru_drive_key(record_id, content)
    signal = np.asarray(content[signal_key], dtype=float).reshape(-1)
    rpm_key = f"X{record_id:03d}RPM"
    rpm = float(np.asarray(content[rpm_key]).reshape(-1)[0]) if rpm_key in content else None
    if signal.size < 2048 or not np.all(np.isfinite(signal)):
        raise RuntimeError(f"invalid signal in {path}")
    return signal, rpm
