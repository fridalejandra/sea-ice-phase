"""
fig_phase_amp_timeseries.py
===========================
Two figures (4 and 5), each with two side-by-side panels (raw vs APAC fitted):

  fig_phase_timeseries.png
      (a) Raw phase anomaly        — 2×3 grid, all sectors + circumpolar
      (b) APAC fitted phase anomaly — 2×3 grid, all sectors + circumpolar

  fig_amplitude_timeseries.png
      (a) Raw amplitude anomaly        — 2×3 grid
      (b) APAC fitted amplitude anomaly — 2×3 grid

All anomalies relative to 1979–2015 baseline.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from scipy.ndimage import uniform_filter1d

from ch3_style import (
    apply_style,
    SECTORS, SECTORS_NO_CIRC,
    SECTOR_COLORS, SECTOR_LABELS,
    DECADE_LEGEND, decade_color,
    zero_line, shade2016, vline2016, stroke,
    save_fig, DEFAULT_OUTPUT_DIR,
)

apply_style()

DATA_DIR   = "/user/geog/falejandraperez/sea-ice-phase/scripts/R/Ch3/data"
OUTPUT_DIR = DEFAULT_OUTPUT_DIR
ANNUAL_CSV = os.path.join(DATA_DIR, "annual_params.csv")

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading data...")
annual = pd.read_csv(ANNUAL_CSV, parse_dates=["min_date", "max_date"])

for col in ["min_doy_anom", "max_doy_anom", "amplitude_anom",
            "amplitude_fitted", "min_doy_fitted", "max_doy_fitted",
            "max_doy_raw", "min_doy_raw", "amplitude_raw_yr",
            "max_doy_raw_anom", "min_doy_raw_anom", "amplitude_raw_anom"]:
    if col in annual.columns:
        annual[col] = pd.to_numeric(annual[col], errors="coerce")

yr_min = int(annual["Year"].min())
yr_max = int(annual["Year"].max())
print(f"  {len(annual)} rows | {yr_min}–{yr_max}")

# ── Baselines ─────────────────────────────────────────────────────────────────
mask_bl = annual["Year"].between(1979, 2015)
bl_doy_median = annual[mask_bl].groupby("sector")["max_doy_raw"].median()
bl_amp_median = annual[mask_bl].groupby("sector")["amplitude_raw_yr"].median()

annual["max_doy_raw_anom_2015"]   = (annual["max_doy_raw"]
                                     - annual["sector"].map(bl_doy_median))
annual["amplitude_raw_anom_2015"] = (annual["amplitude_raw_yr"]
                                     - annual["sector"].map(bl_amp_median))

# ── Sector order ──────────────────────────────────────────────────────────────
PLOT_SECTORS = SECTORS_NO_CIRC + ["SIE_circumpolar"]
PANEL_LETTERS = list("abcdefghijklmnopqrstuvwxyz")


# ── Helper: draw one 2×3 grid of timeseries ───────────────────────────────────
def draw_timeseries_grid(axes_flat, var, ylabel, is_days, letter_offset=0):
    """
    Draw timeseries for all sectors into axes_flat (length 6).
    Returns the post-2016 mean for each sector.
    """
    for idx, (ax, sec) in enumerate(zip(axes_flat, PLOT_SECTORS)):
        sub  = annual[annual["sector"] == sec].sort_values("Year")
        yrs  = sub["Year"].values
        vals = sub[var].values if var in sub.columns else np.full(len(sub), np.nan)

        zero_line(ax)
        vline2016(ax)

        for yr, val in zip(yrs, vals):
            if not np.isnan(val):
                ax.scatter(yr, val, color=decade_color(yr),
                           s=18, zorder=4, edgecolors="white", linewidth=0.3)

        valid = ~np.isnan(vals)
        if valid.sum() >= 5:
            smooth = uniform_filter1d(vals, size=5, mode="nearest")
            ax.plot(yrs, smooth, color=SECTOR_COLORS[sec],
                    lw=1.8, zorder=3, alpha=0.9)

        post_vals = sub[sub["Year"] >= 2016][var] if var in sub.columns else pd.Series([])
        post_mean = post_vals.mean() if len(post_vals) > 0 else np.nan
        if not np.isnan(post_mean):
            label_str = (f"Post-2016: {post_mean:+.1f} d" if is_days
                         else f"Post-2016: {post_mean:+.3f} Mkm²")
            ax.text(0.97, 0.04, label_str, transform=ax.transAxes,
                    fontsize=7, color="#D4537E", ha="right",
                    path_effects=stroke())

        # Panel letter
        ax.text(0.03, 0.97,
                f"({PANEL_LETTERS[letter_offset * 6 + idx]})",
                transform=ax.transAxes, fontsize=8, fontweight="bold",
                va="top", color="#2C2C2A")

        ax.set_title(SECTOR_LABELS[sec], fontsize=9, fontweight="bold",
                     pad=4, color=SECTOR_COLORS[sec])
        ax.tick_params(labelsize=8)
        ax.set_xlim(1977, yr_max + 1)

        # y-label on left column only
        if idx % 3 == 0:
            ax.set_ylabel(ylabel, fontsize=8, labelpad=4)

        # x-label on bottom row only
        if idx >= 3:
            ax.set_xlabel("Year", fontsize=8)

        if sec == "SIE_circumpolar":
            ax.spines["left"].set_linewidth(1.5)
            ax.spines["left"].set_color("#B4B2A9")


# ── Decade legend handles ─────────────────────────────────────────────────────
def decade_legend_handles():
    return [
        plt.scatter([], [], color=c, s=30,
                    edgecolors="white", linewidth=0.4, label=l)
        for c, l in DECADE_LEGEND
    ]


# ── Figure builder ────────────────────────────────────────────────────────────
def make_side_by_side_figure(
    var_raw, var_fitted,
    ylabel_raw, ylabel_fitted,
    title_raw, title_fitted,
    outfile, is_days
):
    """
    One figure with two side-by-side 2×3 panels.
    Left panel  = raw anomaly      (a–f)
    Right panel = APAC fitted anom (g–l)
    """
    fig = plt.figure(figsize=(13.5, 6))

    # Two groups of 2×3 axes side by side with a gap between them
    gs = GridSpec(2, 7, figure=fig,
                  hspace=0.45, wspace=0.25,
                  left=0.06, right=0.98,
                  top=0.88, bottom=0.12,
                  width_ratios=[1, 1, 1, 0.15, 1, 1, 1])

    # Left panel axes (columns 0-2)
    axes_left = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]
    # Right panel axes (columns 4-6, leaving column 3 as gap)
    axes_right = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(4, 7)]

    # Draw grids
    draw_timeseries_grid(axes_left,  var_raw,    ylabel_raw,    is_days, letter_offset=0)
    draw_timeseries_grid(axes_right, var_fitted, ylabel_fitted, is_days, letter_offset=1)

    # Panel group titles
    fig.text(0.24, 0.93, title_raw,    ha="center", fontsize=10,
             fontweight="bold", color="#2C2C2A")
    fig.text(0.74, 0.93, title_fitted, ha="center", fontsize=10,
             fontweight="bold", color="#2C2C2A")

    # Shared legend
    fig.legend(handles=decade_legend_handles(),
               loc="lower center", ncol=5,
               fontsize=7.5, bbox_to_anchor=(0.5, 0.01), frameon=False)

    save_fig(fig, outfile, OUTPUT_DIR)


# ── Phase figure ──────────────────────────────────────────────────────────────
print("\nFig 4: Phase anomaly timeseries (raw vs APAC fitted)")
make_side_by_side_figure(
    var_raw    = "max_doy_raw_anom_2015",
    var_fitted = "max_doy_anom",
    ylabel_raw    = "Phase anomaly (days)\n← Ahead  |  Behind →",
    ylabel_fitted = "Phase anomaly — APAC (days)\n← Ahead  |  Behind →",
    title_raw    = "(a)  Raw observed",
    title_fitted = "(b)  APAC fitted",
    outfile  = "fig04_phase_timeseries.png",
    is_days  = True,
)

# ── Amplitude figure ──────────────────────────────────────────────────────────
print("Fig 5: Amplitude anomaly timeseries (raw vs APAC fitted)")
make_side_by_side_figure(
    var_raw    = "amplitude_raw_anom_2015",
    var_fitted = "amplitude_anom",
    ylabel_raw    = "Amplitude anomaly (million km²)\n← Smaller  |  Larger →",
    ylabel_fitted = "Amplitude anomaly — APAC (million km²)\n← Smaller  |  Larger →",
    title_raw    = "(a)  Raw observed",
    title_fitted = "(b)  APAC fitted",
    outfile  = "fig05_amplitude_timeseries.png",
    is_days  = False,
)

print("\nDone.")