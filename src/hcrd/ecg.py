"""Morphology-based ECG boundary helpers for the QTDB application."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .baselines import gaussian_smooth
from .core import decompose
from .stable import quadratic_curvature_split

GuideKind = Literal["raw", "gaussian", "quadratic"]


@dataclass(frozen=True)
class QRSBoundary:
    onset: int
    fiducial: int
    offset: int


@dataclass(frozen=True)
class DelineationResult:
    onset: int | None
    offset: int | None
    anchor_amplitude: float
    structure_count: int

    @property
    def succeeded(self) -> bool:
        return (
            self.onset is not None
            and self.offset is not None
            and self.onset < self.offset
        )


@dataclass(frozen=True)
class QRSLevelCandidate:
    """One QRS-boundary candidate from one HCRD level and growth threshold."""

    level: int
    amplitude_ratio: float
    onset: int | None
    offset: int | None
    anchor_amplitude: float
    normalized_anchor_amplitude: float
    structure_count: int

    @property
    def succeeded(self) -> bool:
        return (
            self.onset is not None
            and self.offset is not None
            and self.onset < self.offset
        )


def parse_qrs_boundaries(
    samples: ArrayLike,
    symbols: list[str] | tuple[str, ...],
    wave_numbers: ArrayLike,
) -> list[QRSBoundary]:
    """Parse QTDB QRS onset/fiducial/offset triplets.

    QTDB marks waveform onsets and offsets with ``(`` and ``)`` and stores the
    waveform kind in ``num``; value 1 denotes QRS.  Any non-parenthesis symbol
    between the pair is accepted as the QRS fiducial.
    """

    positions = np.asarray(samples, dtype=int)
    numbers = np.asarray(wave_numbers, dtype=int)
    if positions.ndim != 1 or numbers.shape != positions.shape:
        raise ValueError("samples and wave_numbers must be aligned vectors")
    if len(symbols) != positions.size:
        raise ValueError("symbols must align with samples")
    boundaries: list[QRSBoundary] = []
    pending_onset: int | None = None
    pending_fiducial: int | None = None
    for sample, symbol, number in zip(positions, symbols, numbers):
        if symbol == "(" and number == 1:
            pending_onset = int(sample)
            pending_fiducial = None
        elif symbol == ")" and number == 1:
            if (
                pending_onset is not None
                and pending_fiducial is not None
                and pending_onset < pending_fiducial < sample
            ):
                boundaries.append(
                    QRSBoundary(pending_onset, pending_fiducial, int(sample))
                )
            pending_onset = None
            pending_fiducial = None
        elif pending_onset is not None and symbol not in {"(", ")"}:
            pending_fiducial = int(sample)
    return boundaries


def _milliseconds_to_samples(milliseconds: float, sampling_frequency: float) -> int:
    return max(1, int(round(milliseconds * sampling_frequency / 1000.0)))


def hcrd_qrs_multilevel_candidates(
    signal: ArrayLike,
    fiducial: int,
    sampling_frequency: float,
    *,
    guide: GuideKind = "quadratic",
    regularization: float = 100.0,
    gaussian_sigma: float = 2.0,
    pre_window_ms: float = 140.0,
    post_window_ms: float = 180.0,
    anchor_radius_ms: float = 45.0,
    amplitude_ratios: tuple[float, ...] = (0.2,),
    max_levels: int = 4,
) -> tuple[QRSLevelCandidate, ...]:
    """Return boundary candidates from several nested HCRD levels.

    Each level independently anchors its largest structure intersecting the
    fiducial neighbourhood. Adjacent structures are joined for every supplied
    relative-amplitude threshold. The returned candidates are deliberately not
    fused: a rule or learner can combine fine-level localization with coarser
    morphology without changing the HCRD operator itself.
    """

    y = np.asarray(signal, dtype=float)
    if y.ndim != 1 or not np.all(np.isfinite(y)):
        raise ValueError("signal must be a finite one-dimensional array")
    if not 0 <= fiducial < y.size or sampling_frequency <= 0:
        raise ValueError("invalid fiducial or sampling frequency")
    if (
        regularization < 0
        or gaussian_sigma < 0
        or max_levels < 1
        or not amplitude_ratios
        or any(not 0 <= ratio <= 1 for ratio in amplitude_ratios)
    ):
        raise ValueError("invalid guide or amplitude parameter")
    pre = _milliseconds_to_samples(pre_window_ms, sampling_frequency)
    post = _milliseconds_to_samples(post_window_ms, sampling_frequency)
    radius = _milliseconds_to_samples(anchor_radius_ms, sampling_frequency)
    start = max(0, fiducial - pre)
    stop = min(y.size, fiducial + post + 1)
    local = y[start:stop]
    local_fiducial = fiducial - start
    if local.size < 5:
        return ()
    if guide == "raw":
        guided = local
    elif guide == "gaussian":
        guided = gaussian_smooth(local, gaussian_sigma)
    elif guide == "quadratic":
        guided = quadratic_curvature_split(
            local, regularization=regularization
        ).guide
    else:
        raise ValueError(f"unknown guide: {guide}")
    levels = decompose(
        guided,
        atol=1e-12,
        rtol=64 * np.finfo(float).eps,
        max_levels=max_levels,
    ).levels
    guide_scale = max(float(np.sqrt(np.mean(np.square(guided)))), np.finfo(float).eps)
    results: list[QRSLevelCandidate] = []
    for level_index, level in enumerate(levels, start=1):
        knots = level.knots
        if knots.size < 2:
            continue
        amplitudes = np.array(
            [
                np.max(np.abs(level.detail[left : right + 1]))
                for left, right in zip(knots[:-1], knots[1:])
            ]
        )
        distances = np.maximum.reduce(
            [
                knots[:-1] - local_fiducial,
                local_fiducial - knots[1:],
                np.zeros_like(knots[:-1]),
            ]
        )
        anchors = np.flatnonzero(distances <= radius)
        if anchors.size == 0:
            for ratio in amplitude_ratios:
                results.append(
                    QRSLevelCandidate(
                        level_index,
                        ratio,
                        None,
                        None,
                        0.0,
                        0.0,
                        int(amplitudes.size),
                    )
                )
            continue
        anchor = int(anchors[np.argmax(amplitudes[anchors])])
        anchor_amplitude = float(amplitudes[anchor])
        normalized = anchor_amplitude / guide_scale
        for ratio in amplitude_ratios:
            if anchor_amplitude <= np.finfo(float).eps:
                results.append(
                    QRSLevelCandidate(
                        level_index,
                        ratio,
                        None,
                        None,
                        anchor_amplitude,
                        normalized,
                        int(amplitudes.size),
                    )
                )
                continue
            threshold = ratio * anchor_amplitude
            left_structure = anchor
            while (
                left_structure > 0
                and amplitudes[left_structure - 1] >= threshold
            ):
                left_structure -= 1
            right_structure = anchor
            while (
                right_structure + 1 < amplitudes.size
                and amplitudes[right_structure + 1] >= threshold
            ):
                right_structure += 1
            onset = start + int(knots[left_structure])
            offset = start + int(knots[right_structure + 1])
            if not onset < fiducial < offset:
                onset = None
                offset = None
            results.append(
                QRSLevelCandidate(
                    level_index,
                    ratio,
                    onset,
                    offset,
                    anchor_amplitude,
                    normalized,
                    int(amplitudes.size),
                )
            )
    return tuple(results)


def hcrd_qrs_delineate(
    signal: ArrayLike,
    fiducial: int,
    sampling_frequency: float,
    *,
    guide: GuideKind = "quadratic",
    regularization: float = 100.0,
    gaussian_sigma: float = 2.0,
    pre_window_ms: float = 140.0,
    post_window_ms: float = 180.0,
    anchor_radius_ms: float = 45.0,
    amplitude_ratio: float = 0.2,
) -> DelineationResult:
    """Delineate QRS with the frozen first-level anchor-and-grow rule."""

    candidates = hcrd_qrs_multilevel_candidates(
        signal,
        fiducial,
        sampling_frequency,
        guide=guide,
        regularization=regularization,
        gaussian_sigma=gaussian_sigma,
        pre_window_ms=pre_window_ms,
        post_window_ms=post_window_ms,
        anchor_radius_ms=anchor_radius_ms,
        amplitude_ratios=(amplitude_ratio,),
        max_levels=1,
    )
    if not candidates:
        return DelineationResult(None, None, 0.0, 0)
    candidate = candidates[0]
    return DelineationResult(
        candidate.onset,
        candidate.offset,
        candidate.anchor_amplitude,
        candidate.structure_count,
    )


def derivative_qrs_delineate(
    signal: ArrayLike,
    fiducial: int,
    sampling_frequency: float,
    *,
    gaussian_sigma: float = 2.0,
    pre_window_ms: float = 140.0,
    post_window_ms: float = 180.0,
    threshold_ratio: float = 0.15,
) -> DelineationResult:
    """Simple reproducible derivative-threshold comparator."""

    y = np.asarray(signal, dtype=float)
    if y.ndim != 1 or not 0 <= fiducial < y.size:
        raise ValueError("invalid signal or fiducial")
    if not 0 <= threshold_ratio <= 1:
        raise ValueError("threshold_ratio must lie in [0, 1]")
    pre = _milliseconds_to_samples(pre_window_ms, sampling_frequency)
    post = _milliseconds_to_samples(post_window_ms, sampling_frequency)
    start = max(0, fiducial - pre)
    stop = min(y.size, fiducial + post + 1)
    guided = gaussian_smooth(y[start:stop], gaussian_sigma)
    derivative = np.abs(np.gradient(guided))
    local_fiducial = fiducial - start
    radius = _milliseconds_to_samples(50.0, sampling_frequency)
    anchor_left = max(0, local_fiducial - radius)
    anchor_right = min(derivative.size, local_fiducial + radius + 1)
    anchor = anchor_left + int(np.argmax(derivative[anchor_left:anchor_right]))
    peak = float(derivative[anchor])
    threshold = threshold_ratio * peak
    left = anchor
    while left > 0 and derivative[left] >= threshold:
        left -= 1
    right = anchor
    while right + 1 < derivative.size and derivative[right] >= threshold:
        right += 1
    onset = start + left
    offset = start + right
    if not onset < fiducial < offset:
        return DelineationResult(None, None, peak, 1)
    return DelineationResult(onset, offset, peak, 1)
