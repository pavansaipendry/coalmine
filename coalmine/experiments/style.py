"""Shared chart style: the dataviz reference palette, light mode, validated.

Categorical slots are assigned in fixed order (blue = slot 1, green = slot 2)
and never cycled; text wears ink tokens, never series color.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

SERIES_1 = "#2a78d6"  # blue, categorical slot 1
SERIES_2 = "#008300"  # green, categorical slot 2
BAND_1 = "#9ec5f4"  # sequential blue step 200, for uncertainty bands around slot 1


def setup() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "savefig.dpi": 160,
            "axes.edgecolor": BASELINE,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.labelcolor": INK_2,
            "text.color": INK,
        }
    )


def style_axes(ax: plt.Axes) -> None:
    """Recessive chrome: y-gridlines only, bottom spine only, muted ticks."""
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(length=0)


def titles(ax: plt.Axes, title: str, subtitle: str | None = None) -> None:
    pad = 22 + 12 * (subtitle.count("\n") if subtitle else 0)
    ax.set_title(title, loc="left", color=INK, fontsize=12.5, fontweight="bold", pad=pad)
    if subtitle:
        ax.text(
            0.0, 1.04, subtitle, transform=ax.transAxes, color=INK_2, fontsize=9.5, va="bottom"
        )
