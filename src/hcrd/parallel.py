"""Order-preserving batch parallelism for independent HCRD signals.

The hierarchy within one signal is sequential, but different signals or
windows are independent.  A process backend is therefore the appropriate
CPU-parallel implementation for the current pure-Python knot walk; the thread
backend is retained for measurement and for callers whose downstream work
releases the GIL.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from .core import (
    BoundaryRule,
    Decomposition,
    SparseDecomposition,
    decompose,
    decompose_sparse,
)

ParallelBackend = Literal["serial", "thread", "process"]


def _decompose_task(
    task: tuple[
        np.ndarray,
        np.ndarray | None,
        float,
        float,
        BoundaryRule,
        int | None,
        int,
    ],
) -> Decomposition:
    signal, x, atol, rtol, boundary_rule, max_levels, minimum_knot_spacing = task
    return decompose(
        signal,
        x,
        atol=atol,
        rtol=rtol,
        boundary_rule=boundary_rule,
        max_levels=max_levels,
        minimum_knot_spacing=minimum_knot_spacing,
    )


def _decompose_sparse_task(
    task: tuple[
        np.ndarray,
        np.ndarray | None,
        float,
        float,
        BoundaryRule,
        int | None,
        int,
    ],
) -> SparseDecomposition:
    signal, x, atol, rtol, boundary_rule, max_levels, minimum_knot_spacing = task
    return decompose_sparse(
        signal,
        x,
        atol=atol,
        rtol=rtol,
        boundary_rule=boundary_rule,
        max_levels=max_levels,
        minimum_knot_spacing=minimum_knot_spacing,
    )


def decompose_batch(
    signals: list[ArrayLike] | tuple[ArrayLike, ...],
    xs: list[ArrayLike | None] | tuple[ArrayLike | None, ...] | None = None,
    *,
    workers: int = 1,
    backend: ParallelBackend = "process",
    chunksize: int | None = None,
    atol: float = 0.0,
    rtol: float = 64 * np.finfo(float).eps,
    boundary_rule: BoundaryRule = "minimum_curvature",
    max_levels: int | None = None,
    minimum_knot_spacing: int = 1,
) -> tuple[Decomposition, ...]:
    """Decompose independent signals in parallel while preserving input order.

    Process-pool startup and result transfer are included in the call.  For
    small batches, ``backend="serial"`` can therefore be faster.  ``workers``
    is ignored by the serial backend and must be positive in every mode.
    """

    if workers < 1:
        raise ValueError("workers must be positive")
    if backend not in ("serial", "thread", "process"):
        raise ValueError(f"unknown parallel backend: {backend}")

    signal_arrays = [np.asarray(signal, dtype=float) for signal in signals]
    if xs is None:
        x_arrays: list[np.ndarray | None] = [None] * len(signal_arrays)
    else:
        if len(xs) != len(signal_arrays):
            raise ValueError("xs and signals must have the same length")
        x_arrays = [None if x is None else np.asarray(x, dtype=float) for x in xs]

    tasks = [
        (
            signal,
            x,
            atol,
            rtol,
            boundary_rule,
            max_levels,
            minimum_knot_spacing,
        )
        for signal, x in zip(signal_arrays, x_arrays, strict=True)
    ]
    if not tasks:
        return ()
    if backend == "serial" or workers == 1:
        return tuple(_decompose_task(task) for task in tasks)

    effective_chunksize = (
        max(1, len(tasks) // (4 * workers)) if chunksize is None else chunksize
    )
    if effective_chunksize < 1:
        raise ValueError("chunksize must be positive")
    executor_type = ThreadPoolExecutor if backend == "thread" else ProcessPoolExecutor
    with executor_type(max_workers=workers) as executor:
        return tuple(
            executor.map(_decompose_task, tasks, chunksize=effective_chunksize)
        )


def decompose_sparse_batch(
    signals: list[ArrayLike] | tuple[ArrayLike, ...],
    xs: list[ArrayLike | None] | tuple[ArrayLike | None, ...] | None = None,
    *,
    workers: int = 1,
    backend: ParallelBackend = "process",
    chunksize: int | None = None,
    atol: float = 0.0,
    rtol: float = 64 * np.finfo(float).eps,
    boundary_rule: BoundaryRule = "minimum_curvature",
    max_levels: int | None = None,
    minimum_knot_spacing: int = 1,
) -> tuple[SparseDecomposition, ...]:
    """Build independent knot-only hierarchies in input order."""

    if workers < 1:
        raise ValueError("workers must be positive")
    if backend not in ("serial", "thread", "process"):
        raise ValueError(f"unknown parallel backend: {backend}")
    signal_arrays = [np.asarray(signal, dtype=float) for signal in signals]
    if xs is None:
        x_arrays: list[np.ndarray | None] = [None] * len(signal_arrays)
    else:
        if len(xs) != len(signal_arrays):
            raise ValueError("xs and signals must have the same length")
        x_arrays = [None if x is None else np.asarray(x, dtype=float) for x in xs]
    tasks = [
        (
            signal,
            x,
            atol,
            rtol,
            boundary_rule,
            max_levels,
            minimum_knot_spacing,
        )
        for signal, x in zip(signal_arrays, x_arrays, strict=True)
    ]
    if not tasks:
        return ()
    if backend == "serial" or workers == 1:
        return tuple(_decompose_sparse_task(task) for task in tasks)
    effective_chunksize = (
        max(1, len(tasks) // (4 * workers)) if chunksize is None else chunksize
    )
    if effective_chunksize < 1:
        raise ValueError("chunksize must be positive")
    executor_type = ThreadPoolExecutor if backend == "thread" else ProcessPoolExecutor
    with executor_type(max_workers=workers) as executor:
        return tuple(
            executor.map(_decompose_sparse_task, tasks, chunksize=effective_chunksize)
        )
