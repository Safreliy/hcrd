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

    figure, axis = plt.subplots(figsize=(7.2, 8.2))
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_axis_off()

    axis.text(
        0.055,
        0.965,
        "How shape-contrast inversion works",
        ha="left",
        va="top",
        fontsize=17.0,
        fontweight="bold",
        color=DARK,
    )
    axis.text(
        0.055,
        0.923,
        "Local shape evidence removes impossible transition locations.",
        ha="left",
        va="top",
        fontsize=10.6,
        color=MUTED,
    )

    # Step 1: a geometric model of the chord-sign rule.
    _card(axis, 0.655, 0.215)
    _step_badge(axis, 0.096, 0.832, "1")
    axis.text(
        0.135,
        0.842,
        "Find reliable local shape",
        fontsize=12.2,
        fontweight="bold",
        color=DARK,
        va="center",
    )
    axis.text(
        0.135,
        0.798,
        "Compare with endpoint chords on many fixed windows; calibrate all windows together for noise.",
        fontsize=8.35,
        color=TEXT,
        va="top",
    )

    curve_x = np.linspace(0.17, 0.88, 500)
    local_x = (curve_x - 0.17) / (0.88 - 0.17)
    curve_y = 0.682 + 0.075 / (1.0 + np.exp(-8.5 * (local_x - 0.5)))
    axis.plot(curve_x, curve_y, color=DARK, lw=2.3, solid_capstyle="round")

    for first, last, colour, label in (
        (45, 190, GREEN, "convex: chord above"),
        (310, 455, BLUE, "concave: chord below"),
    ):
        chord_x = curve_x[first : last + 1]
        chord_y = np.linspace(curve_y[first], curve_y[last], chord_x.size)
        axis.plot(
            chord_x,
            chord_y,
            color=colour,
            lw=2.4,
            solid_capstyle="round",
        )
        axis.fill_between(
            chord_x,
            curve_y[first : last + 1],
            chord_y,
            color=colour,
            alpha=0.18,
        )
        axis.scatter(
            [curve_x[first], curve_x[last]],
            [curve_y[first], curve_y[last]],
            s=21,
            color=colour,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        _pill(axis, float(np.mean(chord_x)), 0.767, label, colour)

    axis.text(
        0.525,
        0.671,
        "dark line: curve     coloured line: chord",
        ha="center",
        va="top",
        fontsize=8.2,
        color=MUTED,
    )

    _down_arrow(axis, 0.642, 0.612)

    # Step 2: a finite/discrete model of the one-sided logical exclusions.
    _card(axis, 0.365, 0.225)
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
    green_y, blue_y = 0.470, 0.395

    axis.text(
        0.53,
        0.492,
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
        0.417,
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
    _down_arrow(axis, 0.352, 0.322)

    # Step 3: the intersection of all surviving candidate locations.
    _card(axis, 0.105, 0.195)
    _step_badge(axis, 0.096, 0.262, "3")
    axis.text(
        0.135,
        0.272,
        "Keep what the data cannot rule out",
        fontsize=12.2,
        fontweight="bold",
        color=DARK,
        va="center",
    )
    axis.text(
        0.135,
        0.235,
        "Intersect the surviving locations from every certified window.",
        fontsize=9.25,
        color=TEXT,
        va="top",
    )

    final_x0, final_x1 = 0.20, 0.86
    final_left, final_right = 0.405, 0.705
    final_y = 0.168
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
        0.302,
        final_y,
        "ruled out",
        ha="center",
        va="center",
        fontsize=8.4,
        color="#64748B",
    )
    axis.text(
        0.555,
        final_y,
        "95% SCI confidence set",
        ha="center",
        va="center",
        fontsize=9.1,
        fontweight="bold",
        color="white",
    )
    axis.text(
        0.782,
        final_y,
        "ruled out",
        ha="center",
        va="center",
        fontsize=8.4,
        color="#64748B",
    )
    axis.text(
        final_left,
        0.128,
        r"$\widehat L$",
        ha="center",
        va="top",
        fontsize=9.3,
        color=GREEN,
    )
    axis.text(
        final_right,
        0.128,
        r"$\widehat U$",
        ha="center",
        va="top",
        fontsize=9.3,
        color=BLUE,
    )

    axis.text(
        0.5,
        0.045,
        "This picture explains the logic. Finite-sample coverage comes from the theorem, not from the picture.",
        ha="center",
        va="center",
        fontsize=8.4,
        color=MUTED,
    )

    _save(figure)
    print("wrote the three-step sci_method_overview.pdf/png")


if __name__ == "__main__":
    main()
