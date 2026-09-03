"""Minimal shape-contrast inversion example with unknown Gaussian scale."""

from __future__ import annotations

import numpy as np

from hcrd import (
    build_shape_contrast_family,
    gaussian_block_upper_scale,
    gaussian_bonferroni_shape_band,
    invert_s_shaped_inflection,
)


def main() -> None:
    rng = np.random.default_rng(20260903)
    x = np.arange(1, 501, dtype=float) / 501.0
    mean = x - (x - 0.5) ** 3
    y = mean + rng.normal(0.0, 0.01, size=x.size)

    # Spend 0.01 failure probability on an upper noise-scale bound and the
    # remaining 0.04 on the simultaneous shape-contrast band.
    scale = gaussian_block_upper_scale(y, 250, failure_probability=0.01)
    family = build_shape_contrast_family(
        x, separation_multipliers=(1, 2, 4)
    )
    band = gaussian_bonferroni_shape_band(
        family, y, noise_scale=scale.upper_scale, alpha=0.04
    )
    confidence_set = invert_s_shaped_inflection(
        family, band, domain=(0.0, 1.0)
    )

    print(f"upper noise scale: {scale.upper_scale:.4f}")
    print(f"95% SCI set: {confidence_set.interval}")


if __name__ == "__main__":
    main()
