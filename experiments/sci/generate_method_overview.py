"""Generate the executable four-panel overview of shape-contrast inversion.

The figure uses the public ``shapecontrast`` implementation for every
statistical quantity in panels 1, 3, and 4.  Panel 2 is a geometric schematic
of the sign rule used by those contrasts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from shapecontrast import (  # noqa: E402
    build_shape_contrast_family,
    gaussian_bonferroni_shape_band,
    invert_s_shaped_inflection,
)


FIGURE_DIR = PROJECT / "paper" / "sci" / "figures"

GREEN = "#009E73"
BLUE = "#0072B2"
ORANGE = "#E69F00"
GRAY = "#6B7280"
LIGHT_GRAY = "#E5E7EB"
DARK = "#202124"


def _save(figure: plt.Figure) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for suffix, options in (
        ("pdf", {}),
        ("png", {"dpi": 300}),
    ):
        figure.savefig(
            FIGURE_DIR / f"sci_method_overview.{suffix}",
            bbox_inches="tight",
            facecolor="white",
            **options,
        )
    plt.close(figure)


def _style_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=8.4)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.2,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.0,
            "legend.fontsize": 7.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    # A deterministic example with a known transition.  The signal is used
    # only to explain the algorithm; the paper's evidence comes from the
    # frozen benchmark and real-data experiments.
    seed = 20260903
    rng = np.random.default_rng(seed)
    observation_count = 241
    true_transition = 0.56
    noise_scale = 0.02
    x = np.linspace(0.0, 1.0, observation_count)
    mean = 1.0 / (1.0 + np.exp(-12.0 * (x - true_transition)))
    y = mean + rng.normal(0.0, noise_scale, size=x.size)

    family = build_shape_contrast_family(
        x,
        block_sizes=(4, 8, 16, 32, 64),
        separation_multipliers=(1, 2),
    )
    band = gaussian_bonferroni_shape_band(
        family,
        y,
        noise_scale=noise_scale,
        alpha=0.05,
    )
    confidence_set = invert_s_shaped_inflection(
        family,
        band,
        domain=(0.0, 1.0),
    )
    if confidence_set.interval is None:
        raise RuntimeError("the deterministic illustration unexpectedly returned an empty set")
    left, right = confidence_set.interval
    if confidence_set.active_left_contrast is None or confidence_set.active_right_contrast is None:
        raise RuntimeError("the deterministic illustration needs two active contrasts")

    support_left = family.support_left
    support_right = family.support_right
    block_size = family.block_size
    separation = family.separation
    support_midpoint = 0.5 * (support_left + support_right)
    active_left = confidence_set.active_left_contrast
    active_right = confidence_set.active_right_contrast

    figure = plt.figure(figsize=(10.8, 6.4))
    grid = figure.add_gridspec(
        2,
        2,
        left=0.065,
        right=0.985,
        bottom=0.085,
        top=0.92,
        wspace=0.27,
        hspace=0.42,
    )

    # Panel 1: observations and the two contrasts that set the final endpoints.
    axis = figure.add_subplot(grid[0, 0])
    axis.scatter(
        x[::2],
        y[::2],
        s=9,
        color=GRAY,
        alpha=0.38,
        linewidth=0,
        label="noisy observations",
        zorder=2,
    )
    axis.plot(x, mean, color=DARK, lw=2.1, label="mean curve", zorder=3)
    axis.axvline(true_transition, color=ORANGE, lw=1.5, ls=(0, (4, 3)))
    axis.text(
        true_transition + 0.012,
        0.08,
        r"true transition $m_0$",
        color="#A65F00",
        fontsize=8.2,
        rotation=90,
        va="bottom",
    )
    transform = axis.get_xaxis_transform()
    axis.plot(
        [support_left[active_left], support_right[active_left]],
        [0.08, 0.08],
        transform=transform,
        color=GREEN,
        lw=7,
        alpha=0.8,
        solid_capstyle="round",
        clip_on=False,
    )
    axis.plot(
        [support_left[active_right], support_right[active_right]],
        [0.02, 0.02],
        transform=transform,
        color=BLUE,
        lw=7,
        alpha=0.8,
        solid_capstyle="round",
        clip_on=False,
    )
    axis.text(
        0.02,
        0.14,
        "green: convex evidence   blue: concave evidence",
        transform=axis.transAxes,
        fontsize=7.6,
        color="0.32",
    )
    axis.set(xlim=(0.0, 1.0), xlabel="design coordinate", ylabel="response")
    axis.set_title("1  Test local shape on many fixed windows", loc="left", fontweight="bold")
    axis.legend(frameon=False, loc="upper left", handlelength=1.9)
    _style_axis(axis)

    # Panel 2: exact geometric reason for the contrast signs.
    container = figure.add_subplot(grid[0, 1])
    container.set_axis_off()
    container.set_title("2  A chord residual has a safe sign", loc="left", fontweight="bold")
    u = np.linspace(0.0, 1.0, 201)
    for inset_left, sign, colour, label in (
        (0.035, 1.0, GREEN, r"convex $\Rightarrow Q>0$"),
        (0.535, -1.0, BLUE, r"concave $\Rightarrow Q<0$"),
    ):
        inset = container.inset_axes([inset_left, 0.18, 0.43, 0.68])
        curve = sign * (u - 0.5) ** 2
        chord = np.full_like(u, sign * 0.25)
        inset.plot(u, curve, color=DARK, lw=2.0)
        inset.plot(u, chord, color=colour, lw=1.8)
        inset.fill_between(u, curve, chord, color=colour, alpha=0.20)
        inset.scatter([0.0, 0.5, 1.0], [curve[0], curve[100], curve[-1]], s=22, color=DARK, zorder=3)
        inset.vlines(
            0.5,
            min(curve[100], chord[100]),
            max(curve[100], chord[100]),
            color=colour,
            lw=2.0,
        )
        inset.text(0.5, 0.93, label, transform=inset.transAxes, ha="center", va="top", color=colour, fontsize=8.7, fontweight="bold")
        inset.set_xticks([0.0, 0.5, 1.0], labels=["L", "M", "R"])
        inset.set_yticks([])
        inset.spines[["top", "right", "left"]].set_visible(False)
        inset.tick_params(axis="x", length=0, pad=2, labelsize=8)
    container.text(
        0.5,
        0.02,
        r"$Q=$ chord value at $M$ $-$ curve value at $M$; the code averages such residuals over three blocks.",
        ha="center",
        va="bottom",
        fontsize=7.9,
        color="0.28",
        wrap=True,
    )

    # Panel 3: actual simultaneous intervals from one readable subset of rows.
    axis = figure.add_subplot(grid[1, 0])
    scale_rows = np.flatnonzero((block_size == 16) & (separation == 32))
    selected = np.unique(np.concatenate((scale_rows, [active_left, active_right])))
    selected = selected[np.argsort(support_midpoint[selected])]
    ratios = band.estimate[selected] / band.radius[selected]
    signs = band.certified_signs[selected]
    vertical_offsets = np.zeros(selected.size)
    for index in range(1, selected.size):
        if abs(support_midpoint[selected[index]] - support_midpoint[selected[index - 1]]) < 0.012:
            vertical_offsets[index - 1] -= 0.009
            vertical_offsets[index] += 0.009
    y_positions = support_midpoint[selected] + vertical_offsets
    for row, ratio, sign_value, y_position in zip(selected, ratios, signs, y_positions, strict=True):
        colour = GREEN if sign_value > 0 else BLUE if sign_value < 0 else GRAY
        linewidth = 2.8 if row in (active_left, active_right) else 1.7
        marker = "*" if row in (active_left, active_right) else "o"
        marker_size = 62 if marker == "*" else 20
        axis.hlines(y_position, ratio - 1.0, ratio + 1.0, color=colour, lw=linewidth, alpha=0.95)
        axis.scatter([ratio], [y_position], s=marker_size, marker=marker, color=colour, edgecolor="white", linewidth=0.55, zorder=3)
    axis.axvspan(-1.0, 1.0, color=LIGHT_GRAY, alpha=0.55, zorder=0)
    axis.axvline(0.0, color="0.20", lw=1.0, ls=(0, (3, 3)))
    axis.text(0.0, 0.965, "interval crosses zero", transform=axis.get_xaxis_transform(), ha="center", va="top", fontsize=7.5, color="0.38")
    axis.annotate(
        r"sets $\widehat L$",
        xy=(band.estimate[active_left] / band.radius[active_left], support_midpoint[active_left] - 0.009),
        xytext=(2.4, 0.27),
        arrowprops={"arrowstyle": "->", "color": GREEN, "lw": 1.0},
        color=GREEN,
        fontsize=8,
    )
    axis.annotate(
        r"sets $\widehat U$",
        xy=(band.estimate[active_right] / band.radius[active_right], support_midpoint[active_right] + 0.009),
        xytext=(-5.4, 0.76),
        arrowprops={"arrowstyle": "->", "color": BLUE, "lw": 1.0},
        color=BLUE,
        fontsize=8,
    )
    axis.set(
        xlim=(-6.0, 6.0),
        ylim=(0.12, 0.89),
        xlabel="contrast estimate / simultaneous radius",
        ylabel="support midpoint",
    )
    axis.set_title("3  Certify all contrast signs together", loc="left", fontweight="bold")
    axis.legend(
        handles=[
            Line2D([0], [0], color=GREEN, lw=2, marker="o", label="certified positive"),
            Line2D([0], [0], color=GRAY, lw=2, marker="o", label="not certified"),
            Line2D([0], [0], color=BLUE, lw=2, marker="o", label="certified negative"),
        ],
        frameon=False,
        loc="lower left",
        ncol=3,
        handlelength=1.4,
        columnspacing=0.8,
    )
    _style_axis(axis)

    # Panel 4: logical inversion of the two active sign statements.
    axis = figure.add_subplot(grid[1, 1])
    axis.set_title("4  Invert signs into the remaining set", loc="left", fontweight="bold")
    axis.plot([0.0, 1.0], [2.25, 2.25], color=LIGHT_GRAY, lw=11, solid_capstyle="butt")
    axis.plot([0.0, left], [2.25, 2.25], color=GREEN, lw=11, alpha=0.72, solid_capstyle="butt")
    axis.scatter([left], [2.25], s=42, color=GREEN, edgecolor="white", zorder=3)
    axis.text(0.0, 2.56, r"$Q>0$: exclude candidates before $a_T$", fontsize=8.4, color=GREEN)
    axis.plot([0.0, 1.0], [1.45, 1.45], color=LIGHT_GRAY, lw=11, solid_capstyle="butt")
    axis.plot([right, 1.0], [1.45, 1.45], color=BLUE, lw=11, alpha=0.72, solid_capstyle="butt")
    axis.scatter([right], [1.45], s=42, color=BLUE, edgecolor="white", zorder=3)
    axis.text(0.0, 1.76, r"$Q<0$: exclude candidates after $b_T$", fontsize=8.4, color=BLUE)
    axis.plot([0.0, left], [0.50, 0.50], color="#D1D5DB", lw=16, solid_capstyle="butt")
    axis.plot([left, right], [0.50, 0.50], color=ORANGE, lw=16, solid_capstyle="butt")
    axis.plot([right, 1.0], [0.50, 0.50], color="#D1D5DB", lw=16, solid_capstyle="butt")
    axis.vlines([left, right], 0.29, 0.71, colors=[GREEN, BLUE], lw=1.7)
    axis.axvline(true_transition, color="#A65F00", lw=1.3, ls=(0, (4, 3)), ymin=0.06, ymax=0.91)
    axis.text(true_transition, 0.03, r"$m_0$", ha="center", va="bottom", color="#A65F00", fontsize=8.5)
    axis.text(
        0.5 * (left + right),
        0.88,
        rf"SCI $=[{left:.3f},\,{right:.3f}]$",
        ha="center",
        color="#8A5200",
        fontsize=10,
        fontweight="bold",
    )
    axis.text(left, 0.18, r"$\widehat L$", ha="center", va="top", color=GREEN, fontsize=8.7)
    axis.text(right, 0.18, r"$\widehat U$", ha="center", va="top", color=BLUE, fontsize=8.7)
    axis.set(xlim=(-0.02, 1.02), ylim=(-0.03, 2.87), xlabel="candidate transition location")
    axis.set_yticks([])
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(labelsize=8.4)

    figure.suptitle(
        "Shape-contrast inversion: from local geometry to an honest transition set",
        x=0.065,
        ha="left",
        fontsize=13.2,
        fontweight="bold",
    )
    _save(figure)
    print(
        "wrote sci_method_overview.pdf/png; "
        f"M={family.contrast_count}, critical={band.critical_value:.3f}, "
        f"SCI=[{left:.6f}, {right:.6f}]"
    )


if __name__ == "__main__":
    main()
