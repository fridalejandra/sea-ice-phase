"""
fig_phase_amp_independance.py

Two figures:
  1. fig_phase_amplitude_independence_all.png   — all 5 sectors, full page
  2. fig_phase_amplitude_independence_key.png   — Weddell, ABS, East Antarctica

No titles. Panel letters. Detrended.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, linregress

from ch3_style import (
    apply_style,
    SECTORS_NO_CIRC, SECTOR_COLORS, SECTOR_LABELS,
    DECADE_LEGEND, decade_color,
    stroke, save_fig, DEFAULT_OUTPUT_DIR,
)

apply_style()

DATA_DIR   = "/user/geog/falejandraperez/sea-ice-phase/scripts/R/Ch3/data"
OUTPUT_DIR = DEFAULT_OUTPUT_DIR
ANNUAL_CSV = os.path.join(DATA_DIR, "annual_params.csv")
DETREND    = True

print("Loading annual params...")
annual = pd.read_csv(ANNUAL_CSV)
for col in ["max_doy_raw_anom", "amplitude_raw_anom",
            "max_doy_anom", "amplitude_anom"]:
    annual[col] = pd.to_numeric(annual[col], errors="coerce")
print(f"  {len(annual)} rows | {annual['Year'].min()}–{annual['Year'].max()}")


def detrend_series(years, values):
    mask = ~np.isnan(values)
    if mask.sum() < 5:
        return values
    slope, intercept, *_ = linregress(years[mask].astype(float),
                                      values[mask])
    return values - (slope * years.astype(float) + intercept)


def prepare_sector(df, sector, x_col, y_col):
    sub = (df[df["sector"] == sector]
           .sort_values("Year")
           .dropna(subset=[x_col, y_col])
           .copy())
    if DETREND:
        sub[x_col] = detrend_series(sub["Year"].values, sub[x_col].values)
        sub[y_col] = detrend_series(sub["Year"].values, sub[y_col].values)
    return sub


COLS = [
    ("max_doy_raw_anom",  "amplitude_raw_anom",
     "Phase anomaly — raw observed (days)",  "Raw observed"),
    ("max_doy_anom",      "amplitude_anom",
     "Phase anomaly — APAC fitted (days)",   "APAC modelled"),
]

LETTERS = list("abcdefghijklmnopqrst")


def draw_scatter_grid(sectors, figsize, outfile):
    n_rows = len(sectors)
    fig, axes = plt.subplots(n_rows, 2, figsize=figsize, sharey="row")
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    letter_idx = 0
    for row, sector in enumerate(sectors):
        color = SECTOR_COLORS[sector]

        for col, (x_col, y_col, xlabel, col_title) in enumerate(COLS):
            ax  = axes[row, col]
            sub = prepare_sector(annual, sector, x_col, y_col)
            x     = sub[x_col].values
            y     = sub[y_col].values
            years = sub["Year"].values

            for yr, xi, yi in zip(years, x, y):
                ax.scatter(xi, yi, color=decade_color(yr),
                           s=35, zorder=4, edgecolors="white", linewidth=0.3)

            for yr, xi, yi in zip(years, x, y):
                if yr >= 2016:
                    ax.text(xi, yi, str(yr), fontsize=6.5,
                            color="#D4537E", ha="left", va="bottom",
                            path_effects=stroke(lw=2))

            if len(x) > 2:
                m, b   = np.polyfit(x, y, 1)
                x_line = np.linspace(x.min(), x.max(), 100)
                ax.plot(x_line, m * x_line + b,
                        color="#B4B2A9", lw=1.0, ls="--", zorder=3)

            ax.axhline(0, color="grey", lw=0.5, ls="--", zorder=1)
            ax.axvline(0, color="grey", lw=0.5, ls="--", zorder=1)

            r,   p_r   = pearsonr(x, y)
            rho, p_rho = spearmanr(x, y)
            sig = "*" if p_r < 0.05 else ("." if p_r < 0.10 else "")

            ax.text(0.97, 0.97,
                    f"r = {r:+.2f}{sig}   \u03c1 = {rho:+.2f}",
                    transform=ax.transAxes, fontsize=8,
                    ha="right", va="top", color=color,
                    fontweight="bold", path_effects=stroke())

            # Panel letter — top left
            ax.text(0.03, 0.97, f"({LETTERS[letter_idx]})",
                    transform=ax.transAxes, fontsize=8,
                    fontweight="bold", va="top", color="#2C2C2A")
            letter_idx += 1

            ax.set_xlabel(xlabel, fontsize=8)
            ax.set_ylabel("Amplitude anomaly (million km\u00b2)", fontsize=8)
            ax.tick_params(labelsize=8)

            # Sector row label on left column
            if col == 0:
                ax.text(-0.18, 0.5, SECTOR_LABELS[sector],
                        transform=ax.transAxes, fontsize=8,
                        fontweight="bold", color=color,
                        va="center", ha="right", rotation=90)

            # Column titles on top row only
            if row == 0:
                ax.set_title(col_title, fontsize=9,
                             fontweight="bold", pad=6)

    # Decade legend
    handles = [
        plt.scatter([], [], color=c, s=35,
                    edgecolors="white", linewidth=0.3, label=l)
        for c, l in DECADE_LEGEND
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5,
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.04))

    fig.tight_layout(rect=[0.06, 0.05, 1, 1])
    save_fig(fig, outfile, OUTPUT_DIR)


# --- Figure 1: all 5 sectors -----------------------------------------------
print("\nFigure 1: all 5 sectors")
draw_scatter_grid(
    sectors = SECTORS_NO_CIRC,
    figsize = (6.5, 11.0),
    outfile = "fig_phase_amplitude_independence_all.png",
)

# --- Figure 2: 3 key sectors -----------------------------------------------
print("Figure 2: 3 key sectors")
KEY_SECTORS = [
    "SIE_Weddell",
    "SIE_Amundsen_Bellingshausen",
    "SIE_East_Antarctica",
]
draw_scatter_grid(
    sectors = KEY_SECTORS,
    figsize = (6.5, 7.0),
    outfile = "fig_phase_amplitude_independence_key.png",
)

print("\nBoth independence figures saved.")