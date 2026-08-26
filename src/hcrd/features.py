"""Fixed-length component features for real-data representation studies."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .baselines import gaussian_smooth
from .core import decompose
from .robust import adaptive_gaussian_guided_decompose

COMPONENT_FEATURE_NAMES = (
    "log_energy",
    "rms",
    "absolute_mean",
    "standard_deviation",
    "skewness",
    "excess_kurtosis",
    "crest_factor",
    "zero_crossing_rate",
    "spectral_centroid",
    "spectral_entropy",
)


def normalise_window(signal: ArrayLike) -> NDArray[np.float64]:
    """Median-centre and RMS-scale one finite signal window."""

    y = np.asarray(signal, dtype=float)
    if y.ndim != 1 or y.size < 8 or not np.all(np.isfinite(y)):
        raise ValueError("signal must be a finite one-dimensional window of length >= 8")
    centred = y - np.median(y)
    scale = float(np.sqrt(np.mean(centred**2)))
    if scale <= np.finfo(float).eps:
        return np.zeros_like(centred)
    return centred / scale


def component_features(component: ArrayLike) -> NDArray[np.float64]:
    """Return scale-aware morphology and spectrum features for one component."""

    values = np.asarray(component, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("component must be one-dimensional with length >= 2")
    energy = float(np.mean(values**2))
    rms = float(np.sqrt(energy))
    absolute_mean = float(np.mean(np.abs(values)))
    centred = values - np.mean(values)
    standard_deviation = float(np.sqrt(np.mean(centred**2)))
    if standard_deviation <= np.finfo(float).eps:
        skewness = 0.0
        kurtosis = 0.0
    else:
        standardised = centred / standard_deviation
        skewness = float(np.mean(standardised**3))
        kurtosis = float(np.mean(standardised**4) - 3.0)
    crest_factor = float(np.max(np.abs(values)) / max(rms, np.finfo(float).eps))
    signs = np.signbit(values)
    zero_crossing_rate = float(np.mean(signs[1:] != signs[:-1]))
    spectrum = np.abs(np.fft.rfft(values)) ** 2
    spectrum_sum = float(np.sum(spectrum))
    if spectrum_sum <= np.finfo(float).eps:
        spectral_centroid = 0.0
        spectral_entropy = 0.0
    else:
        probabilities = spectrum / spectrum_sum
        frequencies = np.linspace(0.0, 1.0, spectrum.size)
        spectral_centroid = float(np.sum(frequencies * probabilities))
        positive = probabilities > 0
        spectral_entropy = float(
            -np.sum(probabilities[positive] * np.log(probabilities[positive]))
            / np.log(max(2, spectrum.size))
        )
    return np.asarray(
        [
            np.log10(energy + 1e-15),
            rms,
            absolute_mean,
            standard_deviation,
            skewness,
            kurtosis,
            crest_factor,
            zero_crossing_rate,
            spectral_centroid,
            spectral_entropy,
        ],
        dtype=float,
    )


def _pad_components(
    components: list[NDArray[np.float64]], n: int, length: int
) -> list[NDArray[np.float64]]:
    trimmed = components[:n]
    while len(trimmed) < n:
        trimmed.append(np.zeros(length, dtype=float))
    return trimmed


def raw_components(signal: ArrayLike, *, n_components: int = 5) -> list[NDArray[np.float64]]:
    y = normalise_window(signal)
    return _pad_components([y], n_components, y.size)


def gaussian_pyramid_components(
    signal: ArrayLike, *, n_components: int = 5
) -> list[NDArray[np.float64]]:
    y = normalise_window(signal)
    current = y
    components: list[NDArray[np.float64]] = []
    for sigma in (1.0, 2.0, 4.0, 8.0)[: max(0, n_components - 1)]:
        smooth = gaussian_smooth(y, sigma)
        components.append(current - smooth)
        current = smooth
    components.append(current)
    return _pad_components(components, n_components, y.size)


def wavelet_components(
    signal: ArrayLike, *, n_components: int = 5
) -> list[NDArray[np.float64]]:
    import pywt

    y = normalise_window(signal)
    levels = max(1, n_components - 1)
    coefficients = pywt.wavedec(y, "db4", level=levels, mode="symmetric")
    components: list[NDArray[np.float64]] = []
    # Reconstruct fine-to-coarse details, then the coarsest approximation.
    for coefficient_index in range(len(coefficients) - 1, 0, -1):
        isolated = [np.zeros_like(item) for item in coefficients]
        isolated[coefficient_index] = coefficients[coefficient_index]
        reconstructed = pywt.waverec(isolated, "db4", mode="symmetric")[: y.size]
        components.append(np.asarray(reconstructed, dtype=float))
    approximation = [np.zeros_like(item) for item in coefficients]
    approximation[0] = coefficients[0]
    components.append(
        np.asarray(pywt.waverec(approximation, "db4", mode="symmetric")[: y.size], dtype=float)
    )
    return _pad_components(components, n_components, y.size)


def hcrd_components(
    signal: ArrayLike, *, n_components: int = 5
) -> list[NDArray[np.float64]]:
    y = normalise_window(signal)
    result = decompose(y, max_levels=max(1, n_components - 1))
    components = [level.detail.copy() for level in result.levels]
    components.append(result.trend.copy())
    return _pad_components(components, n_components, y.size)


def guided_hcrd_components(
    signal: ArrayLike, *, n_components: int = 5
) -> list[NDArray[np.float64]]:
    y = normalise_window(signal)
    result = adaptive_gaussian_guided_decompose(y)
    components = [result.guided.guide_residual.copy()]
    allowed_details = max(0, n_components - 2)
    components.extend(
        level.detail.copy() for level in result.guided.decomposition.levels[:allowed_details]
    )
    remainder = result.guided.decomposition.trend.copy()
    for level in result.guided.decomposition.levels[allowed_details:]:
        remainder += level.detail
    components.append(remainder)
    return _pad_components(components, n_components, y.size)


def emd_components(signal: ArrayLike, *, n_components: int = 5) -> list[NDArray[np.float64]]:
    from PyEMD import EMD

    y = normalise_window(signal)
    algorithm = EMD()
    imfs = algorithm.emd(y)
    components = [np.asarray(imf, dtype=float) for imf in imfs[: n_components - 1]]
    remainder = y - np.sum(components, axis=0) if components else y.copy()
    components.append(np.asarray(remainder, dtype=float))
    return _pad_components(components, n_components, y.size)


REPRESENTATIONS: dict[str, Callable[..., list[NDArray[np.float64]]]] = {
    "raw": raw_components,
    "gaussian_pyramid": gaussian_pyramid_components,
    "wavelet_db4": wavelet_components,
    "emd": emd_components,
    "hcrd": hcrd_components,
    "hcrd_guided": guided_hcrd_components,
}


def representation_features(
    signal: ArrayLike, representation: str, *, n_components: int = 5
) -> NDArray[np.float64]:
    """Flatten the common feature map over a named component representation."""

    if representation not in REPRESENTATIONS:
        raise ValueError(f"unknown representation: {representation}")
    components = REPRESENTATIONS[representation](signal, n_components=n_components)
    features = np.concatenate([component_features(component) for component in components])
    if not np.all(np.isfinite(features)):
        raise RuntimeError(f"nonfinite features produced by {representation}")
    return features


def _representation_features_task(
    task: tuple[NDArray[np.float64], str, int],
) -> NDArray[np.float64]:
    signal, representation, n_components = task
    return representation_features(
        signal, representation, n_components=n_components
    )


def representation_features_batch(
    signals: Sequence[ArrayLike],
    representation: str,
    *,
    n_components: int = 5,
    workers: int = 1,
    backend: Literal["serial", "thread", "process"] = "process",
    chunksize: int | None = None,
) -> NDArray[np.float64]:
    """Extract independent-window features in input order.

    The process backend bypasses the CPython GIL in the HCRD knot walk.  The
    thread backend is useful only when the chosen representation spends most
    of its time in compiled code that releases the GIL.
    """

    if representation not in REPRESENTATIONS:
        raise ValueError(f"unknown representation: {representation}")
    if workers < 1:
        raise ValueError("workers must be positive")
    if backend not in ("serial", "thread", "process"):
        raise ValueError(f"unknown parallel backend: {backend}")
    tasks = [
        (np.asarray(signal, dtype=float), representation, n_components)
        for signal in signals
    ]
    if not tasks:
        return np.empty((0, n_components * len(COMPONENT_FEATURE_NAMES)), dtype=float)
    if backend == "serial" or workers == 1:
        rows = [_representation_features_task(task) for task in tasks]
    else:
        effective_chunksize = (
            max(1, len(tasks) // (4 * workers)) if chunksize is None else chunksize
        )
        if effective_chunksize < 1:
            raise ValueError("chunksize must be positive")
        executor_type = (
            ThreadPoolExecutor if backend == "thread" else ProcessPoolExecutor
        )
        with executor_type(max_workers=workers) as executor:
            rows = list(
                executor.map(
                    _representation_features_task,
                    tasks,
                    chunksize=effective_chunksize,
                )
            )
    return np.vstack(rows)
