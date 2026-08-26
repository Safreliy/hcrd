#!/usr/bin/env python3
"""Generate the optional vector and 300 dpi graphical abstract."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from hcrd.core import decompose  # noqa: E402


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

BLUE = "#0072B2"
TEAL = "#009E73"
ORANGE = "#D55E00"
MAGENTA = "#CC79A7"
PURPLE = "#6A3D9A"
DARK = "#20252B"
MUTED = "#59636E"
PALE_BLUE = "#EAF3F8"


def _flow_arrow(canvas: plt.Axes, start: float, end: float) -> None:
    """Draw an arrow entirely inside the gutter between adjacent panels."""

    canvas.add_patch(
        FancyArrowPatch(
            (start, 0.505),
            (end, 0.505),
            arrowstyle="-|>",
            mutation_scale=17,
            linewidth=1.8,
            color="#75818D",
            transform=canvas.transAxes,
            clip_on=False,
        )
    )


def _clean_axis(axis: plt.Axes) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    axis.spines[["top", "right", "bottom", "left"]].set_visible(False)


def main() -> None:
    # Use the operator itself to locate every displayed boundary.  This avoids
    # hand-drawn convexity zones drifting away from the implemented method.
    x = np.linspace(0.0, 1.0, 129)
    signal = (
        0.12
        + 0.18 * x
        + 1.40 * np.exp(-0.5 * ((x - 0.43) / 0.075) ** 2)
        + 0.52 * np.exp(-0.5 * ((x - 0.61) / 0.045) ** 2)
        - 0.10 * np.exp(-0.5 * ((x - 0.78) / 0.060) ** 2)
    )
    result = decompose(signal, x, atol=0.0, rtol=0.0, max_levels=2)
    level_1, level_2 = result.levels

    # Elsevier's requested 2.5:1 canvas.  Saving without a tight crop preserves
    # 3984 x 1593 pixels at 300 dpi, above the 1328 x 531 minimum.
    fig = plt.figure(figsize=(13.28, 5.31), facecolor="white")
    canvas = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    canvas.set_axis_off()

    canvas.text(
        0.5,
        0.925,
        "HCRD: curvature-run knots  →  signed chord lobes  →  exact hierarchy",
        ha="center",
        va="center",
        fontsize=20,
        color=DARK,
        fontweight="bold",
        transform=canvas.transAxes,
    )

    axes = [
        fig.add_axes([0.035, 0.285, 0.265, 0.44]),
        fig.add_axes([0.365, 0.285, 0.265, 0.44]),
        fig.add_axes([0.695, 0.285, 0.265, 0.44]),
    ]
    _flow_arrow(canvas, 0.312, 0.353)
    _flow_arrow(canvas, 0.642, 0.683)

    # 1. The coloured pieces are the actual level-1 structures returned by
    # decompose(), and every vertical guide is an actual retained knot.
    axis = axes[0]
    axis.plot(x, signal, color="#C3C9CF", linewidth=1.0, zorder=1)
    for structure in level_1.structures:
        colour = TEAL if structure.sign > 0 else MAGENTA
        sl = slice(structure.left, structure.right + 1)
        axis.plot(
            x[sl],
            signal[sl],
            color=colour,
            linewidth=3.0,
            solid_capstyle="round",
            zorder=2,
        )
    for knot in level_1.knots[1:-1]:
        axis.axvline(x[knot], color="#9AA4AE", linewidth=0.8, linestyle=(0, (2, 2)))
    axis.scatter(
        x[level_1.knots],
        signal[level_1.knots],
        s=34,
        facecolor="white",
        edgecolor=BLUE,
        linewidth=1.3,
        zorder=4,
    )
    _clean_axis(axis)

    # 2. Each structure is replaced by its endpoint chord; the shaded signed
    # residual is the first detail component.
    axis = axes[1]
    positive = signal >= level_1.baseline
    axis.fill_between(
        x,
        level_1.baseline,
        signal,
        where=positive,
        color=TEAL,
        alpha=0.30,
        interpolate=True,
    )
    axis.fill_between(
        x,
        level_1.baseline,
        signal,
        where=~positive,
        color=MAGENTA,
        alpha=0.30,
        interpolate=True,
    )
    axis.plot(x, signal, color="#8C969F", linewidth=1.35)
    axis.plot(x, level_1.baseline, color=ORANGE, linewidth=2.5)
    axis.scatter(
        x[level_1.knots],
        signal[level_1.knots],
        s=31,
        facecolor="white",
        edgecolor=ORANGE,
        linewidth=1.2,
        zorder=4,
    )
    _clean_axis(axis)

    # 3. The same operation is applied only to retained knots.  The coarser
    # baseline and second detail make the nested recursion visible.
    axis = axes[2]
    positive = level_1.baseline >= level_2.baseline
    axis.fill_between(
        x,
        level_2.baseline,
        level_1.baseline,
        where=positive,
        color="#56B4E9",
        alpha=0.32,
        interpolate=True,
    )
    axis.fill_between(
        x,
        level_2.baseline,
        level_1.baseline,
        where=~positive,
        color=MAGENTA,
        alpha=0.27,
        interpolate=True,
    )
    axis.plot(x, level_1.baseline, color="#858F99", linewidth=1.45)
    axis.plot(x, level_2.baseline, color=PURPLE, linewidth=2.6)
    axis.scatter(
        x[level_2.knots],
        level_1.baseline[level_2.knots],
        s=34,
        facecolor="white",
        edgecolor=PURPLE,
        linewidth=1.3,
        zorder=4,
    )
    _clean_axis(axis)

    headings = (
        (0.1675, "1  Find maximal curvature-sign runs", "sign changes become retained knots"),
        (0.4975, "2  Join each run's endpoints", r"$y=b^1+d^1$; shading is the signed lobe $d^1$"),
        (0.8275, "3  Recurse on retained knots", r"$y=b^2+d^2+d^1$ exactly"),
    )
    for center, heading, subheading in headings:
        canvas.text(
            center,
            0.775,
            heading,
            ha="center",
            va="center",
            fontsize=13.4,
            color=DARK,
            fontweight="bold",
            transform=canvas.transAxes,
        )
        canvas.text(
            center,
            0.215,
            subheading,
            ha="center",
            va="center",
            fontsize=10.8,
            color=MUTED,
            transform=canvas.transAxes,
        )

    canvas.add_patch(
        plt.Rectangle(
            (0.035, 0.055),
            0.925,
            0.095,
            facecolor=PALE_BLUE,
            edgecolor="none",
            transform=canvas.transAxes,
            zorder=-1,
        )
    )
    canvas.text(
        0.055,
        0.102,
        "Finite  •  interpretable  •  nested  •  linear total knot-only work",
        ha="left",
        va="center",
        fontsize=12.2,
        color=BLUE,
        fontweight="bold",
        transform=canvas.transAxes,
    )
    canvas.text(
        0.94,
        0.102,
        "Validated for cross-study LC–MS peak curation",
        ha="right",
        va="center",
        fontsize=11.3,
        color=DARK,
        transform=canvas.transAxes,
    )

    fig.savefig(HERE / "graphical_abstract.pdf", facecolor="white")
    fig.savefig(HERE / "graphical_abstract.png", dpi=300, facecolor="white")
    svg_path = HERE / "graphical_abstract.svg"
    fig.savefig(svg_path, facecolor="white")
    svg_text = "\n".join(
        line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()
    )
    svg_path.write_text(
        svg_text + "\n", encoding="utf-8", newline="\n"
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
