"""Full multichannel and structure-set representations of an HCRD hierarchy."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .core import decompose_sparse
from .energy import (
    ENERGY_FEATURE_NAMES_PER_LEVEL,
    level_energies,
    sparse_structure_energies,
)
from .features import normalise_window, wavelet_components


@dataclass(frozen=True)
class HCRDRepresentation:
    """Dense hierarchy channels and a fixed vector describing its structures."""

    channels: NDArray[np.float64]
    structure_features: NDArray[np.float64]
    geometry_features: NDArray[np.float64]


def _safe_quantiles(values: NDArray[np.float64]) -> NDArray[np.float64]:
    if values.size == 0:
        return np.zeros(5, dtype=float)
    return np.quantile(values, [0.1, 0.25, 0.5, 0.75, 0.9])


def _level_energy_row(item: object | None, sample_count: int) -> list[float]:
    if item is None:
        return [0.0] * len(ENERGY_FEATURE_NAMES_PER_LEVEL)
    total_area = float(item.polygon_area)
    positive_fraction = (
        float(item.positive_polygon_area) / total_area if total_area > 0.0 else 0.0
    )
    sign_balance = (
        (float(item.positive_polygon_area) - float(item.negative_polygon_area))
        / total_area
        if total_area > 0.0
        else 0.0
    )
    return [
        float(np.log1p(item.structure_count)),
        float(item.knot_count / sample_count),
        float(np.log1p(item.polygon_area / max(1, sample_count - 1))),
        float(np.log1p(item.quadratic_energy / max(1, sample_count - 1))),
        float(np.log1p(item.triangle_area / max(1, sample_count - 1))),
        float(np.log1p(item.peak_amplitude)),
        float(np.log1p(item.mean_amplitude)),
        positive_fraction,
        sign_balance,
        float(item.area_concentration),
        float(item.weighted_shape_factor),
        float(item.mean_duration / max(1, sample_count - 1)),
    ]


def structure_feature_names(
    *, max_levels: int = 5, spatial_bins: int = 8, top_k: int = 8
) -> tuple[str, ...]:
    """Stable names for :func:`hcrd_representation` structure features."""

    if max_levels < 1 or spatial_bins < 1 or top_k < 1:
        raise ValueError("max_levels, spatial_bins and top_k must be positive")
    names: list[str] = []
    for level in range(max_levels):
        prefix = f"level_{level}"
        names.extend(f"{prefix}_energy_{name}" for name in ENERGY_FEATURE_NAMES_PER_LEVEL)
        names.extend(
            [
                f"{prefix}_positive_count_fraction",
                f"{prefix}_negative_count_fraction",
            ]
        )
        for quantity in ("duration", "amplitude", "polygon_area", "quadratic_energy", "shape_factor"):
            names.extend(
                f"{prefix}_{quantity}_q{quantile}"
                for quantile in (10, 25, 50, 75, 90)
            )
        for spatial_bin in range(spatial_bins):
            names.extend(
                [
                    f"{prefix}_bin_{spatial_bin}_count_fraction",
                    f"{prefix}_bin_{spatial_bin}_signed_area_fraction",
                    f"{prefix}_bin_{spatial_bin}_peak_amplitude",
                ]
            )
        for rank in range(top_k):
            names.extend(
                f"{prefix}_top_{rank}_{quantity}"
                for quantity in (
                    "midpoint",
                    "duration",
                    "sign",
                    "amplitude",
                    "polygon_area",
                    "quadratic_energy",
                    "shape_factor",
                )
            )
    return tuple(names)


def geometry_feature_names(*, max_levels: int = 5) -> tuple[str, ...]:
    """Training-only descriptors allowed in discovery subgroup search."""

    names: list[str] = ["depth_fraction"]
    for level in range(max_levels):
        names.extend(
            [
                f"level_{level}_knot_fraction",
                f"level_{level}_structure_density",
                f"level_{level}_area_concentration",
                f"level_{level}_sign_balance",
                f"level_{level}_mean_duration_fraction",
                f"level_{level}_detail_energy_fraction",
            ]
        )
    return tuple(names)


def hcrd_representation(
    signal: ArrayLike,
    *,
    max_levels: int = 5,
    spatial_bins: int = 8,
    top_k: int = 8,
) -> HCRDRepresentation:
    """Return complete HCRD channels, structure bank and geometry descriptors."""

    if max_levels < 1 or spatial_bins < 1 or top_k < 1:
        raise ValueError("max_levels, spatial_bins and top_k must be positive")
    y = normalise_window(signal)
    sample_count = y.size
    duration_scale = float(max(1, sample_count - 1))
    sparse = decompose_sparse(y, max_levels=max_levels)
    dense = sparse.materialize()
    details = [level.detail.copy() for level in dense.levels]
    details.extend(
        np.zeros(sample_count, dtype=float)
        for _ in range(max_levels - len(details))
    )
    channels = np.stack([*details, dense.trend.copy()])
    # aeon rejects any case/channel pair whose standard deviation is <= 1e-7.
    # A finite HCRD hierarchy legitimately has absent (exactly zero) detail
    # channels.  Give only such channels a label-independent numerical carrier
    # and cancel it in another channel, preserving exact reconstruction.
    carrier_grid = np.linspace(-1.0, 1.0, sample_count)
    for channel_index in range(channels.shape[0]):
        if float(np.std(channels[channel_index])) > 1e-7:
            continue
        carrier = 1e-4 * np.roll(carrier_grid, channel_index)
        cancel_index = channels.shape[0] - 1 if channel_index == 0 else 0
        channels[channel_index] += carrier
        channels[cancel_index] -= carrier

    energy_levels = level_energies(sparse)
    structure_levels = sparse_structure_energies(sparse)
    rows: list[float] = []
    geometry: list[float] = [float(sparse.depth / max_levels)]
    signal_energy = float(np.mean(y**2))
    for level_index in range(max_levels):
        energy_item = (
            energy_levels[level_index] if level_index < len(energy_levels) else None
        )
        structures = (
            structure_levels[level_index]
            if level_index < len(structure_levels)
            else ()
        )
        rows.extend(_level_energy_row(energy_item, sample_count))
        positive_count = sum(item.sign > 0 for item in structures)
        negative_count = sum(item.sign < 0 for item in structures)
        rows.extend(
            [
                positive_count / sample_count,
                negative_count / sample_count,
            ]
        )

        durations = np.asarray(
            [item.duration / duration_scale for item in structures], dtype=float
        )
        amplitudes = np.asarray([item.amplitude for item in structures], dtype=float)
        polygon_areas = np.asarray(
            [item.polygon_area / duration_scale for item in structures], dtype=float
        )
        quadratic = np.asarray(
            [item.quadratic_energy / duration_scale for item in structures], dtype=float
        )
        shape_factors = np.asarray(
            [item.shape_factor for item in structures], dtype=float
        )
        for values in (durations, amplitudes, polygon_areas, quadratic, shape_factors):
            rows.extend(_safe_quantiles(values).tolist())

        total_area = float(np.sum(polygon_areas))
        bin_rows = np.zeros((spatial_bins, 3), dtype=float)
        for item, area in zip(structures, polygon_areas, strict=True):
            midpoint = 0.5 * (item.left + item.right) / duration_scale
            index = min(spatial_bins - 1, int(midpoint * spatial_bins))
            bin_rows[index, 0] += 1.0 / sample_count
            bin_rows[index, 1] += item.sign * area / max(
                total_area, np.finfo(float).eps
            )
            bin_rows[index, 2] = max(bin_rows[index, 2], item.amplitude)
        rows.extend(bin_rows.ravel().tolist())

        ranked = sorted(structures, key=lambda item: item.polygon_area, reverse=True)
        for rank in range(top_k):
            if rank >= len(ranked):
                rows.extend([0.0] * 7)
                continue
            item = ranked[rank]
            rows.extend(
                [
                    0.5 * (item.left + item.right) / duration_scale,
                    item.duration / duration_scale,
                    float(item.sign),
                    item.amplitude,
                    item.polygon_area / duration_scale,
                    item.quadratic_energy / duration_scale,
                    item.shape_factor,
                ]
            )

        if energy_item is None:
            geometry.extend([0.0] * 6)
        else:
            detail_energy = float(np.mean(dense.levels[level_index].detail**2))
            area = float(energy_item.polygon_area)
            sign_balance = (
                (energy_item.positive_polygon_area - energy_item.negative_polygon_area)
                / area
                if area > 0.0
                else 0.0
            )
            geometry.extend(
                [
                    energy_item.knot_count / sample_count,
                    energy_item.structure_count / sample_count,
                    energy_item.area_concentration,
                    sign_balance,
                    energy_item.mean_duration / duration_scale,
                    detail_energy / max(signal_energy, np.finfo(float).eps),
                ]
            )

    feature_array = np.asarray(rows, dtype=float)
    geometry_array = np.asarray(geometry, dtype=float)
    if feature_array.size != len(
        structure_feature_names(
            max_levels=max_levels, spatial_bins=spatial_bins, top_k=top_k
        )
    ):
        raise RuntimeError("structure feature layout mismatch")
    if geometry_array.size != len(geometry_feature_names(max_levels=max_levels)):
        raise RuntimeError("geometry feature layout mismatch")
    if not (
        np.all(np.isfinite(channels))
        and np.all(np.isfinite(feature_array))
        and np.all(np.isfinite(geometry_array))
    ):
        raise RuntimeError("nonfinite HCRD representation")
    return HCRDRepresentation(channels, feature_array, geometry_array)


def _hcrd_representation_task(
    task: tuple[NDArray[np.float64], int, int, int],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    signal, max_levels, spatial_bins, top_k = task
    representation = hcrd_representation(
        signal, max_levels=max_levels, spatial_bins=spatial_bins, top_k=top_k
    )
    return (
        representation.channels,
        representation.structure_features,
        representation.geometry_features,
    )


def hcrd_representation_batch(
    signals: Sequence[ArrayLike],
    *,
    max_levels: int = 5,
    spatial_bins: int = 8,
    top_k: int = 8,
    workers: int = 1,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Process independent series in parallel and preserve their input order."""

    if workers < 1:
        raise ValueError("workers must be positive")
    tasks = [
        (np.asarray(signal, dtype=float), max_levels, spatial_bins, top_k)
        for signal in signals
    ]
    if workers == 1:
        results = [_hcrd_representation_task(task) for task in tasks]
    else:
        chunksize = max(1, len(tasks) // (4 * workers))
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(
                executor.map(_hcrd_representation_task, tasks, chunksize=chunksize)
            )
    if not results:
        feature_count = len(
            structure_feature_names(
                max_levels=max_levels, spatial_bins=spatial_bins, top_k=top_k
            )
        )
        geometry_count = len(geometry_feature_names(max_levels=max_levels))
        return (
            np.empty((0, max_levels + 1, 0), dtype=float),
            np.empty((0, feature_count), dtype=float),
            np.empty((0, geometry_count), dtype=float),
        )
    channels, features, geometry = zip(*results, strict=True)
    return np.stack(channels), np.stack(features), np.stack(geometry)


def raw_channel_batch(signals: Sequence[ArrayLike]) -> NDArray[np.float64]:
    """Return normalized signals with one collection channel."""

    return np.stack([normalise_window(signal) for signal in signals])[:, None, :]


def wavelet_channel_batch(
    signals: Sequence[ArrayLike], *, max_levels: int = 5
) -> NDArray[np.float64]:
    """Return the equal-channel-count db4 control representation."""

    return np.stack(
        [wavelet_components(signal, n_components=max_levels + 1) for signal in signals]
    )
