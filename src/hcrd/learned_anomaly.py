"""Small label-free learners over the full HCRD component representation."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.ensemble import IsolationForest

from .anomaly import aggregate_area_density
from .components import multiscale_detail_series
from .temporal_anomaly import empirical_rank


def _forest_score(
    features: NDArray[np.float64], train_size: int
) -> NDArray[np.float64]:
    model = IsolationForest(
        n_estimators=128,
        max_samples=min(512, train_size),
        contamination="auto",
        max_features=1.0,
        bootstrap=False,
        random_state=20240825,
        n_jobs=1,
    )
    model.fit(features[:train_size])
    return -model.score_samples(features)


def hcrd_component_iforest_scores(
    signal: ArrayLike,
    *,
    train_size: int,
    max_levels: int = 8,
) -> dict[str, NDArray[np.float64]]:
    """Return the fixed A3 semisupervised component-forest candidate family."""

    values = np.asarray(signal, dtype=float)
    if values.ndim != 1 or values.size < 4 or not np.all(np.isfinite(values)):
        raise ValueError("signal must be a finite one-dimensional array")
    if not 4 <= train_size <= values.size:
        raise ValueError("train_size must be between 4 and signal length")
    details = multiscale_detail_series(values, max_levels=max_levels)
    area = np.abs(details)
    total = np.sum(area, axis=0)
    direct = aggregate_area_density(area, aggregation="max")

    mass = np.sum(area, axis=0)
    fractions = np.divide(
        area,
        mass[None, :],
        out=np.zeros_like(area),
        where=mass[None, :] > 0.0,
    )
    levels = np.arange(1, details.shape[0] + 1, dtype=float)[:, None]
    barycentre = np.sum(fractions * levels, axis=0)
    safe_fractions = np.where(fractions > 0.0, fractions, 1.0)
    entropy = -np.sum(fractions * np.log(safe_fractions), axis=0)
    difference = np.diff(values, prepend=values[0])
    raw_features = np.column_stack((values, difference))
    signed_features = details.T
    area_features = area.T
    signed_area_features = np.column_stack((signed_features, area_features))
    full_features = np.column_stack(
        (
            signed_features,
            area_features,
            raw_features,
            total,
            barycentre,
            entropy,
        )
    )

    base = {
        "a3_if_raw": _forest_score(raw_features, train_size),
        "a3_if_signed": _forest_score(signed_features, train_size),
        "a3_if_area": _forest_score(area_features, train_size),
        "a3_if_signed_area": _forest_score(signed_area_features, train_size),
        "a3_if_full": _forest_score(full_features, train_size),
    }
    rank_direct = empirical_rank(direct)
    output: dict[str, NDArray[np.float64]] = {"a3_direct": direct, **base}
    for name, score in base.items():
        output[f"a3_fuse_direct_{name.removeprefix('a3_')}"] = (
            0.5 * rank_direct + 0.5 * empirical_rank(score)
        )
    return output
