"""
fig_phase_amp_timeseries.py
Phase and amplitude timeseries, rolling variance, and variability summaries.

Panels:
  3a. Phase anomaly timeseries      — 2×3 grid, all sectors + circumpolar
  3b. Amplitude anomaly timeseries  — 2×3 grid, all sectors + circumpolar
  3c. Selected sectors combined     — Weddell, Ross, King Haakon
  3d. Rolling variance              — with pre/post-2016 mean lines
  3e. Pre/post-2016 variability     — grouped bars
  3f. Overall variability by sector — three bar panels

All anomalies relative to 1979–2015 baseline.
Titles removed — use as figure captions.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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

# --- Load ------------------------------------------------------------------
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

# --- Baselines -------------------------------------------------------------
mask_bl = annual["Year"].between(1979, 2015)

bl_doy_median = annual[mask_bl].groupby("sector")["max_doy_raw"].median()
bl_amp_median = annual[mask_bl].groupby("sector")["amplitude_raw_yr"].median()
bl_min_median = annual[mask_bl].groupby("sector")["min_doy_raw"].median()
bl_doy_std    = annual[mask_bl].groupby("sector")["max_doy_raw"].std()
bl_amp_std    = annual[mask_bl].groupby("sector")["amplitude_raw_yr"].std()

annual["max_doy_raw_anom_2015"]   = (annual["max_doy_raw"]
                                     - annual["sector"].map(bl_doy_median))
annual["amplitude_raw_anom_2015"] = (annual["amplitude_raw_yr"]
                                     - annual["sector"].map(bl_amp_median))
annual["min_doy_raw_anom_2015"]   = (annual["min_doy_raw"]
                                     - annual["sector"].map(bl_min_median))
annual["max_doy_raw_anom_2015_z"] = (annual["max_doy_raw_anom_2015"]
                                     / annual["sector"].map(bl_doy_std))
annual["amplitude_raw_anom_2015_z"] = (annual["amplitude_raw_anom_2015"]
                                       / annual["sector"].map(bl_amp_std))

# --- Legend helper ---------------------------------------------------------
def decade_legend_handles(smooth_color="#2C2C2A", smooth_lw=1.8):
    handles = [
        plt.scatter([], [], color=c, s=30,
                    edgecolors="white", linewidth=0.4, label=l)
        for c, l in DECADE_LEGEND
    ]
    if smooth_color:
        handles.append(
            Line2D([0], [0], color=smooth_color, lw=smooth_lw,
                   label="5-yr running mean")
        )
    return handles

PANEL_LETTERS = list("abcdefghijklmnopqrstuvwxyz")

# =============================================================================
# PANELS 3a & 3b — timeseries 2×3 grid
# =============================================================================

def plot_anomaly_timeseries(var, ylabel, outfile, is_days=True):
    plot_sectors = SECTORS_NO_CIRC + ["SIE_circumpolar"]
    # 2 rows × 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(6.5, 5.0),
                             sharey=False, sharex=True)
    axes_flat = axes.flatten()

    for idx, (ax, sec) in enumerate(zip(axes_flat, plot_sectors)):
        sub  = annual[annual["sector"] == sec].sort_values("Year")
        yrs  = sub["Year"].values
        vals = sub[var].values

        zero_line(ax)
        vline2016(ax)

        for yr, val in zip(yrs, vals):
            ax.scatter(yr, val, color=decade_color(yr),
                       s=18, zorder=4, edgecolors="white", linewidth=0.3)

        if len(vals) >= 5:
            smooth = uniform_filter1d(vals, size=5, mode="nearest")
            ax.plot(yrs, smooth, color=SECTOR_COLORS[sec],
                    lw=1.8, zorder=3, alpha=0.9)

        post_mean = sub[sub["Year"] >= 2016][var].mean()
        label_str = (f"Post-2016: {post_mean:+.1f} d" if is_days
                     else f"Post-2016: {post_mean:+.3f} Mkm\u00b2")
        ax.text(0.97, 0.04, label_str, transform=ax.transAxes,
                fontsize=7, color="#D4537E", ha="right",
                path_effects=stroke())

        # Panel letter + sector title
        ax.text(0.03, 0.97, f"({PANEL_LETTERS[idx]})",
                transform=ax.transAxes, fontsize=8, fontweight="bold",
                va="top", color="#2C2C2A")
        ax.set_title(SECTOR_LABELS[sec], fontsize=9, fontweight="bold", pad=4,
                     color=SECTOR_COLORS[sec])
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

    fig.legend(handles=decade_legend_handles(smooth_color=None),
               loc="lower center", ncol=6,
               fontsize=7.5, bbox_to_anchor=(0.5, -0.04), frameon=False)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    save_fig(fig, outfile, OUTPUT_DIR)


print("Panel 3a: phase anomaly timeseries")
plot_anomaly_timeseries(
    var     = "max_doy_raw_anom_2015",
    ylabel  = "Phase anomaly (days)\n\u2190 Ahead  |  Behind \u2192",
    outfile = "fig_phase_amplitude_timeseries_3a_phase.png",
    is_days = True,
)

print("Panel 3b: amplitude anomaly timeseries")
plot_anomaly_timeseries(
    var     = "amplitude_raw_anom_2015",
    ylabel  = "Amplitude anomaly (million km\u00b2)\n\u2190 Smaller  |  Larger \u2192",
    outfile = "fig_phase_amplitude_timeseries_3b_amplitude.png",
    is_days = False,
)

# =============================================================================
# PANEL 3c — selected sectors
# =============================================================================
print("Panel 3c: selected sectors combined")

SELECTED = {
    "SIE_Weddell"    : "#2196F3",
    "SIE_Ross"       : "#4CAF50",
    "SIE_King_Haakon": "#9C27B0",
}

fig, axes = plt.subplots(2, 3, figsize=(6.5, 4.5),
                         sharex=True, sharey="row")

row_vars    = ["max_doy_raw_anom_2015", "amplitude_raw_anom_2015"]
row_ylabels = [
    "Phase anomaly (days)\n\u2190 Ahead  |  Behind \u2192",
    "Amplitude anomaly (million km\u00b2)\n\u2190 Smaller  |  Larger \u2192",
]
letter_idx = 0

for row, (var, ylabel) in enumerate(zip(row_vars, row_ylabels)):
    for col, (sec, color) in enumerate(SELECTED.items()):
        ax  = axes[row, col]
        sub = annual[annual["sector"] == sec].sort_values("Year")
        yrs = sub["Year"].values
        vals= sub[var].values

        zero_line(ax)
        vline2016(ax)

        for yr, val in zip(yrs, vals):
            ax.scatter(yr, val, color=decade_color(yr),
                       s=18, zorder=4, edgecolors="white", linewidth=0.3)

        if len(vals) >= 5:
            smooth = uniform_filter1d(vals, size=5, mode="nearest")
            ax.plot(yrs, smooth, color=color, lw=2.0, zorder=3, alpha=0.9)

        post_mean = sub[sub["Year"] >= 2016][var].mean()
        is_days   = "doy" in var
        ax.text(0.97, 0.04,
                f"Post-2016: {post_mean:+.1f} d" if is_days
                else f"Post-2016: {post_mean:+.3f} Mkm\u00b2",
                transform=ax.transAxes, fontsize=7,
                color="#D4537E", ha="right", path_effects=stroke())

        ax.text(0.03, 0.97, f"({PANEL_LETTERS[letter_idx]})",
                transform=ax.transAxes, fontsize=8, fontweight="bold",
                va="top", color="#2C2C2A")
        letter_idx += 1

        if row == 0:
            ax.set_title(SECTOR_LABELS[sec], fontsize=9,
                         fontweight="bold", pad=4, color=color)
        if col == 0:
            ax.set_ylabel(ylabel, fontsize=8, labelpad=4)
        if row == 1:
            ax.set_xlabel("Year", fontsize=8)

        ax.tick_params(labelsize=8)
        ax.set_xlim(1977, yr_max + 1)

fig.legend(handles=decade_legend_handles(),
           loc="lower center", ncol=7,
           fontsize=7.5, bbox_to_anchor=(0.5, -0.04), frameon=False)
fig.tight_layout(rect=[0, 0.04, 1, 1])
save_fig(fig, "fig_phase_amplitude_timeseries_3c_selected.png", OUTPUT_DIR)


# =============================================================================
# PANEL 3d — rolling variance: all sectors as lines on 2 panels
# =============================================================================
print("Panel 3d: rolling variance")

all_sectors_roll = SECTORS_NO_CIRC + ["SIE_circumpolar"]
row_vars   = ["max_doy_raw_anom_2015", "amplitude_raw_anom_2015"]
row_labels = [
    "(a)  10-yr rolling std dev of phase (days)",
    "(b)  10-yr rolling std dev of amplitude (million km\u00b2)",
]

fig, axes = plt.subplots(2, 1, figsize=(6.5, 5.0), sharex=True)

for ax, var, ylabel in zip(axes, row_vars, row_labels):
    for sec in all_sectors_roll:
        sub  = (annual[annual["sector"] == sec]
                .sort_values("Year").set_index("Year"))
        roll = sub[var].rolling(10, center=True, min_periods=6).std()
        color = SECTOR_COLORS[sec]
        lw    = 2.2 if sec == "SIE_circumpolar" else 1.6
        ls    = "--" if sec == "SIE_circumpolar" else "-"
        ax.plot(roll.index, roll.values, color=color, lw=lw, ls=ls,
                zorder=3, label=SECTOR_LABELS[sec])

    shade2016(ax, yr_max=yr_max)
    ax.axvline(2016, color="#D4537E", lw=0.9, ls=":", alpha=0.7, zorder=4)
    ax.set_ylim(bottom=0)
    ax.set_xlim(yr_min + 4, yr_max - 4)
    ax.set_ylabel(ylabel.split("  ")[1], fontsize=9)
    ax.text(0.02, 0.97, ylabel.split("  ")[0],
            transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="top", color="#2C2C2A")
    ax.tick_params(labelsize=8)

axes[1].set_xlabel("Year", fontsize=9)

# Single legend below both panels
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=6,
           fontsize=8, bbox_to_anchor=(0.5, -0.04), frameon=False)
fig.tight_layout(rect=[0, 0.05, 1, 1])
save_fig(fig, "fig_phase_amplitude_timeseries_3d_rolling_variance.png", OUTPUT_DIR)


# =============================================================================
# PANEL 3e — pre/post-2016 variability bars
# =============================================================================
print("Panel 3e: pre/post-2016 variability bars")

pre  = annual[annual["Year"] <  2016]
post = annual[annual["Year"] >= 2016]

all_sectors = SECTORS_NO_CIRC + ["SIE_circumpolar"]
colors_bar  = [SECTOR_COLORS[s] for s in all_sectors]
labels_bar  = [SECTOR_LABELS[s] for s in all_sectors]
x           = np.arange(len(all_sectors))
width       = 0.35

phase_pre  = [pre[pre["sector"]==s]["max_doy_raw_anom_2015"].dropna().std()
              for s in all_sectors]
phase_post = [post[post["sector"]==s]["max_doy_raw_anom_2015"].dropna().std()
              for s in all_sectors]
amp_pre    = [pre[pre["sector"]==s]["amplitude_raw_anom_2015"].dropna().std()
              for s in all_sectors]
amp_post   = [post[post["sector"]==s]["amplitude_raw_anom_2015"].dropna().std()
              for s in all_sectors]

fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.8))

for panel_idx, (ax, pre_vals, post_vals, ylabel, letter, fmt) in enumerate(zip(
    axes,
    [phase_pre,  amp_pre],
    [phase_post, amp_post],
    ["Std dev of phase anomaly (days)",
     "Std dev of amplitude anomaly (million km\u00b2)"],
    ["(a)", "(b)"],
    ["{:.1f}", "{:.2f}"]
)):
    bars_pre  = ax.bar(x - width/2, pre_vals, width,
                       color=colors_bar, alpha=1.0,
                       edgecolor="white", label="1979\u20132015", zorder=3)
    bars_post = ax.bar(x + width/2, post_vals, width,
                       color=colors_bar, alpha=0.45,
                       edgecolor="white", label="2016\u2013present",
                       hatch="///", zorder=3)

    for bar, val in zip(bars_pre, pre_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(pre_vals) * 0.02,
                fmt.format(val),
                ha="center", va="bottom", fontsize=8,
                fontweight="bold", path_effects=stroke())

    for bar, val in zip(bars_post, post_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(pre_vals) * 0.02,
                fmt.format(val),
                ha="center", va="bottom", fontsize=8,
                color="#5F5E5A", path_effects=stroke())

    ax.set_xticks(x)
    ax.set_xticklabels(labels_bar, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_ylim(0, max(max(pre_vals), max(post_vals)) * 1.35)
    ax.legend(fontsize=8, loc="upper right")
    ax.text(0.02, 0.97, letter, transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top")

fig.text(0.5, -0.04,
         "Note: post-2016 period is shorter \u2014 std dev estimates are less stable",
         ha="center", fontsize=8, color="#5F5E5A", style="italic")
fig.tight_layout()
save_fig(fig, "fig_phase_amplitude_timeseries_3e_prepost_bars.png", OUTPUT_DIR)


# =============================================================================
# PANEL 3f — overall variability by sector
# =============================================================================
print("Panel 3f: overall phase and amplitude variability by sector")

all_sectors_f = SECTORS_NO_CIRC + ["SIE_circumpolar"]
colors_f      = [SECTOR_COLORS[s] for s in all_sectors_f]
labels_f      = [SECTOR_LABELS[s] for s in all_sectors_f]
x_f           = np.arange(len(all_sectors_f))

phase_std_max = [annual[annual["sector"]==s]["max_doy_raw"].dropna().std()
                 for s in all_sectors_f]
phase_std_min = [annual[annual["sector"]==s]["min_doy_raw"].dropna().std()
                 for s in all_sectors_f]
amp_std       = [annual[annual["sector"]==s]["amplitude_raw_yr"].dropna().std()
                 for s in all_sectors_f]

fig, axes = plt.subplots(1, 3, figsize=(6.5, 3.5))

panels_f = [
    (axes[0], phase_std_max, "Melt onset timing",   "Std dev (days)",        "{:.1f}d", "(a)"),
    (axes[1], phase_std_min, "Freeze onset timing",  "Std dev (days)",        "{:.1f}d", "(b)"),
    (axes[2], amp_std,       "Amplitude variability","Std dev (million km\u00b2)", "{:.2f}", "(c)"),
]

for ax, vals, title, ylabel, fmt, letter in panels_f:
    bars = ax.bar(x_f, vals, color=colors_f, width=0.6,
                  edgecolor="white", zorder=3)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vals) * 0.02,
                fmt.format(val),
                ha="center", va="bottom",
                fontsize=8, fontweight="bold", path_effects=stroke())
    ax.set_xticks(x_f)
    ax.set_xticklabels(labels_f, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontweight="bold", fontsize=9, pad=4)
    ax.set_ylim(0, max(vals) * 1.3)
    ax.text(0.02, 0.97, letter, transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top")

fig.tight_layout()
save_fig(fig, "fig_phase_amplitude_timeseries_3f_variability_by_sector.png", OUTPUT_DIR)

print("\nAll fig_phase_amplitude_timeseries panels complete.")