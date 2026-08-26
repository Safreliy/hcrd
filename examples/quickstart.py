"""Small deterministic HCRD example with optional visualization."""

from __future__ import annotations

import argparse

import numpy as np

from hcrd import decompose_sparse, level_energies
from hcrd.signals import alternating_chord_lobes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    example = alternating_chord_lobes(
        seed=20260915,
        intervals=6,
        piecewise_baseline=True,
        amplitude_variation=True,
    )
    sparse = decompose_sparse(example.observed, example.x, atol=0.0, rtol=0.0)
    dense = sparse.materialize()
    error = float(np.max(np.abs(dense.reconstruct() - example.observed)))

    print(f"samples: {example.observed.size}")
    print(f"depth: {sparse.depth}")
    print(f"knots per level: {[knots.size for knots in sparse.knot_sets]}")
    print(f"stored knots: {sparse.stored_knot_count}")
    energy = level_energies(sparse)[0]
    print(f"first-level polygon mass: {energy.polygon_area:.6g}")
    print(f"first-level quadratic energy: {energy.quadratic_energy:.6g}")
    print(f"max reconstruction error: {error:.3e}")
    if error > 1e-12:
        raise RuntimeError("exact reconstruction check failed")

    if args.plot:
        import matplotlib.pyplot as plt

        first = dense.levels[0]
        plt.plot(example.x, example.observed, color="0.2", label="signal")
        plt.plot(example.x, first.baseline, color="#D55E00", label="first baseline")
        plt.fill_between(
            example.x,
            first.baseline,
            example.observed,
            where=first.detail >= 0,
            color="#009E73",
            alpha=0.3,
        )
        plt.fill_between(
            example.x,
            first.baseline,
            example.observed,
            where=first.detail < 0,
            color="#CC79A7",
            alpha=0.3,
        )
        plt.scatter(example.x[first.knots], example.observed[first.knots], s=18)
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
