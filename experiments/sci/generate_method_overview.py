"""Generate a plain-language overview of shape-contrast inversion.

This is deliberately a schematic rather than an experiment plot. It keeps
only the three ideas a reader needs on first contact: local chord signs, the
one-sided exclusions implied by those signs, and their intersection.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


PROJECT = Path(__file__).resolve().parents[2]
FIGURE_DIR = PROJECT / "paper" / "sci" / "figures"

GREEN = "#009E73"
BLUE = "#0072B2"
ORANGE = "#E69F00"
DARK = "#202124"
TEXT = "#374151"
MUTED = "#6B7280"
LIGHT = "#E5E7EB"
CARD_EDGE = "#D1D5DB"
CARD_FILL = "#FAFAFA"


def _save(figure: plt.Figure) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for suffix, options in (("pdf", {}), ("png", {"dpi": 300})):
        figure.savefig(
            FIGURE_DIR / f"sci_method_overview.{suffix}",
            bbox_inches="tight",
            pad_inches=0.08,
            facecolor="white",
            **options,
        )
    plt.close(figure)


def _card(axis: plt.Axes, bottom: float, height: float) -> None:
    axis.add_patch(
        FancyBboxPatch(
            (0.055, bottom),
            0.89,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor=CARD_FILL,
            edgecolor=CARD_EDGE,
            linewidth=1.0,
        )
    )


def _step_badge(axis: plt.Axes, x: float, y: float, number: str) -> None:
    axis.text(
        x,
        y,
        number,
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color="white",
        bbox={
            "boxstyle": "circle,pad=0.30",
            "facecolor": DARK,
            "edgecolor": "none",
        },
    )


def _pill(axis: plt.Axes, x: float, y: float, label: str, colour: str) -> None:
    axis.text(
        x,
        y,
        label,
        ha="center",
        va="center",
        fontsize=7.8,
        fontweight="bold",
        color=colour,
        bbox={
            "boxstyle": "round,pad=0.28,rounding_size=0.45",
            "facecolor": "white",
            "edgecolor": colour,
            "linewidth": 1.15,
        },
    )


def _down_arrow(axis: plt.Axes, top: float, bottom: float) -> None:
    axis.add_patch(
        FancyArrowPatch(
            (0.5, top),
            (0.5, bottom),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.4,
            color="#9CA3AF",
        )
    )


def _cross(axis: plt.Axes, x: float, y: float) -> None:
    dx, dy = 0.009, 0.007
    axis.plot(
        [x - dx, x + dx],
        [y - dy, y + dy],
        color="#9CA3AF",
        lw=1.4,
    )
    axis.plot(
        [x - dx, x + dx],
        [y + dy, y - dy],
        color="#9CA3AF",
        lw=1.4,
    )


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure, axis = plt.subplots(figsize=(7.2, 6.4))
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_axis_off()

    # Step 1: a geometric model of the chord-sign rule.
    _card(axis, 0.650, 0.300)
    _step_badge(axis, 0.096, 0.912, "1")
    axis.text(
        0.135,
        0.922,
        "Find reliable local shape",
        fontsize=12.2,
        fontweight="bold",
        color=DARK,
        va="center",
    )
    axis.text(
        0.135,
        0.875,
        "On many fixed windows, SCI compares the curve with an endpoint chord.\n"
        "The signs are calibrated together to account for noise.",
        fontsize=8.6,
        color=TEXT,
        va="top",
        linespacing=1.35,
    )

    local_coordinate = np.linspace(-1.0, 1.0, 201)
    for centre, sign, colour, label in (
        (0.33, 1.0, GREEN, "convex: chord above"),
        (0.69, -1.0, BLUE, "concave: chord below"),
    ):
        curve_x = centre + 0.135 * local_coordinate
        if sign > 0.0:
            curve_y = 0.680 + 0.062 * local_coordinate**2
            chord_y = np.full_like(curve_x, 0.742)
        else:
            curve_y = 0.680 + 0.062 * (1.0 - local_coordinate**2)
            chord_y = np.full_like(curve_x, 0.680)
        axis.plot(
            curve_x,
            curve_y,
            color=DARK,
            lw=2.4,
            solid_capstyle="round",
        )
        axis.plot(
            curve_x,
            chord_y,
            color=colour,
            lw=2.8,
            solid_capstyle="round",
        )
        axis.fill_between(
            curve_x,
            curve_y,
            chord_y,
            color=colour,
            alpha=0.20,
        )
        axis.scatter(
            [curve_x[0], curve_x[-1]],
            [curve_y[0], curve_y[-1]],
            s=25,
            color=colour,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        axis.scatter(
            [centre],
            [curve_y[curve_y.size // 2]],
            s=22,
            color=DARK,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        _pill(axis, centre, 0.785, label, colour)

    _down_arrow(axis, 0.638, 0.608)

    # Step 2: a finite/discrete model of the one-sided logical exclusions.
    _card(axis, 0.350, 0.240)
    _step_badge(axis, 0.096, 0.552, "2")
    axis.text(
        0.135,
        0.562,
        "Rule out one side",
        fontsize=12.2,
        fontweight="bold",
        color=DARK,
        va="center",
    )
    x0, x1 = 0.20, 0.86
    a_location, b_location = 0.43, 0.68
    green_y, blue_y = 0.455, 0.375

    axis.text(
        0.53,
        0.477,
        r"convex $\Rightarrow$ transition right of $a_T$",
        color=GREEN,
        fontsize=8.8,
        fontweight="bold",
        ha="center",
        va="bottom",
    )
    axis.plot(
        [x0, x1],
        [green_y, green_y],
        color=LIGHT,
        lw=10,
        solid_capstyle="butt",
    )
    axis.plot(
        [a_location, x1],
        [green_y, green_y],
        color=GREEN,
        lw=10,
        alpha=0.72,
        solid_capstyle="butt",
    )
    axis.add_patch(
        FancyArrowPatch(
            (a_location + 0.015, green_y),
            (x1 - 0.005, green_y),
            arrowstyle="-|>",
            mutation_scale=13,
            color=GREEN,
            lw=1.5,
        )
    )
    _cross(axis, 0.275, green_y)
    _cross(axis, 0.355, green_y)
    axis.vlines(
        a_location,
        green_y - 0.023,
        green_y + 0.023,
        color=GREEN,
        lw=1.6,
    )
    axis.text(
        0.53,
        0.397,
        r"concave $\Rightarrow$ transition left of $b_T$",
        color=BLUE,
        fontsize=8.8,
        fontweight="bold",
        ha="center",
        va="bottom",
    )
    axis.plot(
        [x0, x1],
        [blue_y, blue_y],
        color=LIGHT,
        lw=10,
        solid_capstyle="butt",
    )
    axis.plot(
        [x0, b_location],
        [blue_y, blue_y],
        color=BLUE,
        lw=10,
        alpha=0.72,
        solid_capstyle="butt",
    )
    axis.add_patch(
        FancyArrowPatch(
            (b_location - 0.015, blue_y),
            (x0 + 0.005, blue_y),
            arrowstyle="-|>",
            mutation_scale=13,
            color=BLUE,
            lw=1.5,
        )
    )
    _cross(axis, 0.745, blue_y)
    _cross(axis, 0.825, blue_y)
    axis.vlines(
        b_location,
        blue_y - 0.023,
        blue_y + 0.023,
        color=BLUE,
        lw=1.6,
    )
    _down_arrow(axis, 0.337, 0.307)

    # Step 3: the intersection of all surviving candidate locations.
    _card(axis, 0.040, 0.250)
    _step_badge(axis, 0.096, 0.247, "3")
    axis.text(
        0.135,
        0.257,
        "Keep what the data cannot rule out",
        fontsize=12.2,
        fontweight="bold",
        color=DARK,
        va="center",
    )
    axis.text(
        0.135,
        0.205,
        "Intersect the surviving locations from every certified window.",
        fontsize=9.25,
        color=TEXT,
        va="top",
    )

    final_x0, final_x1 = 0.17, 0.87
    final_left, final_right = 0.37, 0.73
    final_y = 0.120
    axis.plot(
        [final_x0, final_left],
        [final_y, final_y],
        color="#CBD5E1",
        lw=17,
        solid_capstyle="butt",
    )
    axis.plot(
        [final_left, final_right],
        [final_y, final_y],
        color=ORANGE,
        lw=17,
        solid_capstyle="butt",
    )
    axis.plot(
        [final_right, final_x1],
        [final_y, final_y],
        color="#CBD5E1",
        lw=17,
        solid_capstyle="butt",
    )
    axis.vlines(
        [final_left, final_right],
        final_y - 0.029,
        final_y + 0.029,
        colors=[GREEN, BLUE],
        lw=2.0,
    )
    axis.text(
        0.270,
        final_y,
        "ruled out",
        ha="center",
        va="center",
        fontsize=8.4,
        color="#64748B",
    )
    axis.text(
        0.550,
        final_y,
        "95% SCI set",
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color="white",
    )
    axis.text(
        0.800,
        final_y,
        "ruled out",
        ha="center",
        va="center",
        fontsize=8.4,
        color="#64748B",
    )
    axis.text(
        final_left,
        0.077,
        r"$\widehat L$",
        ha="center",
        va="top",
        fontsize=9.3,
        color=GREEN,
    )
    axis.text(
        final_right,
        0.077,
        r"$\widehat U$",
        ha="center",
        va="top",
        fontsize=9.3,
        color=BLUE,
    )

    _save(figure)
    print("wrote the three-step sci_method_overview.pdf/png")


if __name__ == "__main__":
    main()
