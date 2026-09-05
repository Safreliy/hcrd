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
    axis.scatter(
        [x],
        [y],
        s=200,
        marker="o",
        color=DARK,
        edgecolor="none",
        zorder=4,
    )
    axis.text(
        x,
        y - 0.001,
        number,
        ha="center",
        va="center",
        fontsize=7.2,
        fontweight="bold",
        color="white",
        zorder=5,
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
    head_height = 0.011
    head_half_width = 0.010
    axis.plot(
        [0.5, 0.5],
        [top, bottom + head_height],
        color="#9CA3AF",
        lw=1.6,
        solid_capstyle="round",
        zorder=3,
    )
    axis.fill(
        [0.5 - head_half_width, 0.5 + head_half_width, 0.5],
        [bottom + head_height, bottom + head_height, bottom],
        color="#9CA3AF",
        zorder=4,
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
    _card(axis, 0.650, 0.330)
    _step_badge(axis, 0.096, 0.940, "1")
    axis.text(
        0.135,
        0.940,
        "Find reliable contrast signs",
        fontsize=12.2,
        fontweight="bold",
        color=DARK,
        va="center",
    )
    axis.text(
        0.135,
        0.905,
        "On many fixed windows, SCI compares the curve with an endpoint chord.\n"
        "The signs are calibrated together to account for noise.",
        fontsize=8.6,
        color=TEXT,
        va="top",
        linespacing=1.35,
    )

    curve_parameter = np.linspace(0.0, 1.0, 501)
    curve_x = 0.17 + 0.71 * curve_parameter
    raw_curve = np.where(
        curve_parameter <= 0.5,
        4.0 * curve_parameter**3,
        1.0 - 4.0 * (1.0 - curve_parameter) ** 3,
    )
    curve_y = 0.655 + 0.125 * raw_curve
    axis.plot(
        curve_x,
        curve_y,
        color=DARK,
        lw=2.7,
        solid_capstyle="round",
        zorder=3,
    )

    for first, last, colour, label in (
        (10, 240, GREEN, "convex lobe: chord above"),
        (260, 490, BLUE, "concave lobe: chord below"),
    ):
        lobe_x = curve_x[first : last + 1]
        lobe_curve = curve_y[first : last + 1]
        chord_y = np.linspace(lobe_curve[0], lobe_curve[-1], lobe_x.size)
        axis.plot(
            lobe_x,
            chord_y,
            color=colour,
            lw=3.0,
            solid_capstyle="round",
            zorder=4,
        )
        axis.fill_between(
            lobe_x,
            lobe_curve,
            chord_y,
            color=colour,
            alpha=0.25,
            zorder=2,
        )
        axis.scatter(
            [lobe_x[0], lobe_x[-1]],
            [lobe_curve[0], lobe_curve[-1]],
            s=26,
            color=colour,
            edgecolor="white",
            linewidth=0.6,
            zorder=5,
        )
        _pill(axis, float(np.mean(lobe_x)), 0.810, label, colour)

    _down_arrow(axis, 0.638, 0.602)

    # Step 2: a finite/discrete model of the one-sided logical exclusions.
    _card(axis, 0.350, 0.240)
    _step_badge(axis, 0.096, 0.552, "2")
    axis.text(
        0.135,
        0.552,
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
        r"positive mean contrast $\Rightarrow$ transition right of $a_T$",
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
        r"negative mean contrast $\Rightarrow$ transition left of $b_T$",
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
    _down_arrow(axis, 0.338, 0.302)

    # Step 3: the intersection of all surviving candidate locations.
    _card(axis, 0.040, 0.250)
    _step_badge(axis, 0.096, 0.247, "3")
    axis.text(
        0.135,
        0.247,
        "Keep the surviving locations",
        fontsize=12.2,
        fontweight="bold",
        color=DARK,
        va="center",
    )
    axis.text(
        0.135,
        0.205,
        "Intersect the sign-based constraints and take the closure.",
        fontsize=9.25,
        color=TEXT,
        va="top",
    )

    # Keep the same coordinate scale: intersection must not widen either bound.
    final_x0, final_x1 = x0, x1
    final_left, final_right = a_location, b_location
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
        (final_x0 + final_left) / 2,
        final_y,
        "ruled out",
        ha="center",
        va="center",
        fontsize=8.4,
        color="#64748B",
    )
    axis.text(
        (final_left + final_right) / 2,
        final_y,
        "95% SCI set",
        ha="center",
        va="center",
        fontsize=8.6,
        fontweight="bold",
        color="white",
    )
    axis.text(
        (final_right + final_x1) / 2,
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
