"""Frozen feature representations for expert-labelled LC--MS EICs."""

from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .core import decompose_sparse
from .energy import level_energies, sparse_structure_energies

EICPartition = Literal["train", "validation", "confirmation"]


@dataclass(frozen=True)
class EICFeatureBank:
    """The five frozen E1 representations for one chromatogram."""

    raw64: NDArray[np.float32]
    domain: NDArray[np.float32]
    hcrd_1: NDArray[np.float32]
    hcrd_8: NDArray[np.float32]
    hcrd_geometry: NDArray[np.float32]
    area_only: NDArray[np.float32]


@dataclass(frozen=True)
class EICPairFeatureBank:
    """Concatenated short- and long-window representations seen by experts."""

    raw64: NDArray[np.float32]
    domain: NDArray[np.float32]
    hcrd_1: NDArray[np.float32]
    hcrd_8: NDArray[np.float32]
    hcrd_geometry: NDArray[np.float32]
    area_only: NDArray[np.float32]


def eic_partition(axis: str, identifier: str) -> EICPartition:
    """Return the frozen double-group partition from the E1 protocol."""

    if axis not in {"sample", "peak"}:
        raise ValueError("axis must be 'sample' or 'peak'")
    normalized = unicodedata.normalize("NFKC", str(identifier)).strip().lower()
    byte = hashlib.sha256(
        f"hcrd-e1-v1|{axis}|{normalized}".encode("utf-8")
    ).digest()[0]
    if byte <= 153:
        return "train"
    if byte <= 204:
        return "validation"
    return "confirmation"


def _sanitize_eic(
    intensity: ArrayLike, retention_time: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    y = np.asarray(intensity, dtype=float).ravel()
    x = np.asarray(retention_time, dtype=float).ravel()
    if x.shape != y.shape:
        raise ValueError("intensity and retention_time must have equal lengths")
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size == 0:
        raise ValueError("EIC has no finite samples")
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    unique, starts = np.unique(x, return_index=True)
    if unique.size != x.size:
        stops = np.r_[starts[1:], x.size]
        y = np.asarray(
            [np.median(y[left:right]) for left, right in zip(starts, stops, strict=True)]
        )
        x = unique
    if x.size < 8:
        raise ValueError("EIC must contain at least 8 distinct retention times")
    return y, x


def _normalise(y: NDArray[np.float64]) -> tuple[NDArray[np.float64], float, float]:
    center = float(np.median(y))
    q05, q95 = np.quantile(y, [0.05, 0.95])
    mad = float(np.median(np.abs(y - center)))
    scale = max(float(q95 - q05), 1.4826 * mad, np.finfo(float).eps)
    return (y - center) / scale, center, scale


def _resample(
    values: NDArray[np.float64], x: NDArray[np.float64], size: int = 64
) -> NDArray[np.float64]:
    grid = np.linspace(x[0], x[-1], size)
    return np.interp(grid, x, values)


def _raw_features(
    y: NDArray[np.float64], x: NDArray[np.float64]
) -> NDArray[np.float64]:
    from scipy.signal import find_peaks, peak_widths

    z, _, _ = _normalise(y)
    duration = max(float(x[-1] - x[0]), np.finfo(float).eps)
    shifted = np.maximum(y - float(np.min(y)), 0.0)
    mass = float(np.trapezoid(shifted, x))
    weights = shifted + np.finfo(float).eps
    center_mass = float(np.sum(weights * x) / np.sum(weights))
    peak_index = int(np.argmax(y))
    peaks, properties = find_peaks(z, prominence=0.0)
    prominences = np.asarray(properties.get("prominences", []), dtype=float)
    max_prominence = float(np.max(prominences, initial=0.0))
    if peaks.size:
        best = int(np.argmax(prominences))
        widths = peak_widths(z, peaks[[best]], rel_height=0.5)
        half_width = float(widths[0][0] / max(1, y.size - 1))
        left_width = float(peaks[best] - widths[2][0])
        right_width = float(widths[3][0] - peaks[best])
        asymmetry = left_width / max(right_width, np.finfo(float).eps)
    else:
        half_width = 0.0
        asymmetry = 0.0
    scalars = np.asarray(
        [
            np.log1p(max(float(np.max(y) - np.min(y)), 0.0)),
            np.log1p(max(mass, 0.0)),
            (center_mass - x[0]) / duration,
            (x[peak_index] - x[0]) / duration,
            half_width,
            np.log1p(max(asymmetry, 0.0)),
            np.mean(np.abs(np.diff(z))),
            np.mean(np.abs(np.diff(z, n=2))),
            np.mean(y == 0.0),
            max_prominence,
            peaks.size / y.size,
        ],
        dtype=float,
    )
    return np.r_[_resample(z, x), scalars]


def _domain_features(
    y: NDArray[np.float64], x: NDArray[np.float64]
) -> NDArray[np.float64]:
    from scipy.ndimage import gaussian_filter1d, gaussian_laplace, white_tophat
    from scipy.signal import find_peaks, peak_widths

    z, _, _ = _normalise(y)
    rows: list[float] = []
    for scale in (1, 2, 4, 8):
        smooth = gaussian_filter1d(z, sigma=scale, mode="nearest")
        log_response = -gaussian_laplace(z, sigma=scale, mode="nearest")
        top_hat = white_tophat(z, size=2 * scale + 1, mode="nearest")
        peaks, properties = find_peaks(smooth, prominence=0.0)
        prominence = np.asarray(properties.get("prominences", []), dtype=float)
        if peaks.size:
            widths = peak_widths(smooth, peaks, rel_height=0.5)[0]
        else:
            widths = np.asarray([], dtype=float)
        rows.extend(
            [
                float(np.max(smooth, initial=0.0)),
                float(np.max(log_response, initial=0.0)),
                float(np.quantile(log_response, 0.95)),
                float(np.max(top_hat, initial=0.0)),
                float(np.mean(np.maximum(top_hat, 0.0))),
                float(np.max(prominence, initial=0.0)),
                float(np.mean(prominence)) if prominence.size else 0.0,
                float(np.max(widths, initial=0.0) / max(1, y.size - 1)),
                float(peaks.size / y.size),
            ]
        )
    return np.asarray(rows, dtype=float)


def _hcrd_features(
    y: NDArray[np.float64], x: NDArray[np.float64], *, max_levels: int = 8
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    _, _, scale = _normalise(y)
    hierarchy = decompose_sparse(y, x, max_levels=max_levels)
    dense = hierarchy.materialize()
    summaries = level_energies(hierarchy)
    structures_by_level = sparse_structure_energies(hierarchy)
    total_duration = max(float(x[-1] - x[0]), np.finfo(float).eps)
    channels: list[float] = []
    geometry: list[float] = [float(hierarchy.depth / max_levels)]
    area_only: list[float] = []
    previous_total_area = 0.0

    for level_index in range(max_levels):
        if level_index >= hierarchy.depth:
            channels.extend([0.0] * 64)
            geometry.extend([0.0] * 37)
            area_only.extend([0.0] * 6)
            continue
        detail = dense.levels[level_index].detail / scale
        channels.extend(_resample(detail, x).tolist())
        summary = summaries[level_index]
        structures = structures_by_level[level_index]
        total_area = float(summary.polygon_area)
        sign_balance = (
            (summary.positive_polygon_area - summary.negative_polygon_area)
            / total_area
            if total_area > 0.0
            else 0.0
        )
        duration_quantiles = (
            np.quantile(
                np.asarray([item.duration for item in structures]) / total_duration,
                [0.25, 0.5, 0.75],
            )
            if structures
            else np.zeros(3)
        )
        geometry.extend(
            [
                1.0,
                summary.structure_count / y.size,
                summary.knot_count / y.size,
                np.log1p(summary.positive_polygon_area),
                np.log1p(summary.negative_polygon_area),
                np.log1p(summary.triangle_area),
                np.log1p(summary.quadratic_energy),
                np.log1p(summary.peak_amplitude),
                summary.area_concentration,
                sign_balance,
                *duration_quantiles.tolist(),
            ]
        )
        ranked = sorted(structures, key=lambda item: item.polygon_area, reverse=True)
        for rank in range(4):
            if rank >= len(ranked):
                geometry.extend([0.0] * 6)
                continue
            item = ranked[rank]
            geometry.extend(
                [
                    (0.5 * (x[item.left] + x[item.right]) - x[0]) / total_duration,
                    item.duration / total_duration,
                    float(item.sign),
                    np.log1p(item.amplitude),
                    np.log1p(item.polygon_area),
                    item.shape_factor,
                ]
            )
        area_only.extend(
            [
                np.log1p(summary.positive_polygon_area),
                np.log1p(summary.negative_polygon_area),
                np.log1p(total_area),
                np.log1p(summary.triangle_area),
                np.log1p(summary.quadratic_energy),
                np.log1p(total_area) - np.log1p(previous_total_area),
            ]
        )
        previous_total_area = total_area

    trend = dense.trend / scale
    channels.extend(_resample(trend, x).tolist())
    return (
        np.asarray(channels, dtype=float),
        np.asarray(geometry, dtype=float),
        np.asarray(area_only, dtype=float),
    )


def eic_feature_bank(
    intensity: ArrayLike, retention_time: ArrayLike
) -> EICFeatureBank:
    """Compute every representation fixed before the E1 data were inspected."""

    y, x = _sanitize_eic(intensity, retention_time)
    raw = _raw_features(y, x)
    domain_extra = _domain_features(y, x)
    channels, geometry, area = _hcrd_features(y, x, max_levels=8)
    first_level_channels = channels[:64]
    first_level_geometry = np.r_[geometry[:1], geometry[1:38]]
    arrays = {
        "raw64": raw,
        "domain": np.r_[raw, domain_extra],
        "hcrd_1": np.r_[raw, first_level_channels, first_level_geometry],
        "hcrd_8": np.r_[raw, channels, geometry],
        "hcrd_geometry": geometry,
        "area_only": area,
    }
    for name, values in arrays.items():
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"nonfinite values in EIC representation {name}")
    return EICFeatureBank(
        **{name: values.astype(np.float32) for name, values in arrays.items()}
    )


def eic_pair_feature_bank(
    short_intensity: ArrayLike,
    short_retention_time: ArrayLike,
    long_intensity: ArrayLike,
    long_retention_time: ArrayLike,
) -> EICPairFeatureBank:
    """Return frozen features for the two EIC windows shown to annotators."""

    short = eic_feature_bank(short_intensity, short_retention_time)
    long = eic_feature_bank(long_intensity, long_retention_time)
    return EICPairFeatureBank(
        **{
            name: np.concatenate([getattr(short, name), getattr(long, name)])
            for name in EICFeatureBank.__dataclass_fields__
        }
    )
