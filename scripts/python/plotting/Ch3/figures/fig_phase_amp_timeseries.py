"""
Phase and amplitude timeseries, variance, and pre/post-2016 variability.

This is the main descriptive figure for Ch3 — it shows what phase and
amplitude actually do over the record before we get into atmospheric drivers.
Broken into sub-panels so each can be used independently if needed.

    3a. Phase anomaly timeseries      — all sectors + circumpolar
    3b. Amplitude anomaly timeseries  — all sectors + circumpolar
    3c. Phase & amplitude combined    — selected sectors (Weddell, Ross, King Haakon)
    3d. Rolling variance              — 10-yr rolling std dev, shows the post-2016 jump
    3e. Pre/post-2016 variability     — grouped bars summarising the variance shift
    3f. Season length                 — tabled for now, code kept at the bottom

All anomalies are relative to the 1979–2015 baseline.
Needs annual_params.csv from compute_phase_amplitude.py.
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

DATA_DIR   = "/user/geog/falejandraperez/sea-ice-phase/scripts/R/chapter3/data"
OUTPUT_DIR = DEFAULT_OUTPUT_DIR
ANNUAL_CSV = os.path.join(DATA_DIR, "annual_params.csv")

# --- Load and clean --------------------------------------------------------

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
# Everything is anomalised relative to 1979–2015 so the post-2016 period
# stands out without the baseline choice being buried in the methods.

mask_bl = annual["Year"].between(1979, 2015)

bl_doy_median = annual[mask_bl].groupby("sector")["max_doy_raw"].median()
bl_amp_median = annual[mask_bl].groupby("sector")["amplitude_raw_yr"].median()
bl_min_median = annual[mask_bl].groupby("sector")["min_doy_raw"].median()

annual["max_doy_raw_anom_2015"]   = (annual["max_doy_raw"]
                                     - annual["sector"].map(bl_doy_median))
annual["amplitude_raw_anom_2015"] = (annual["amplitude_raw_yr"]
                                     - annual["sector"].map(bl_amp_median))
annual["min_doy_raw_anom_2015"]   = (annual["min_doy_raw"]
                                     - annual["sector"].map(bl_min_median))

# Z-scores using pre-2016 std dev — puts phase (days) and amplitude (Mkm²)
# on the same scale for the case study figures
bl_doy_std = annual[mask_bl].groupby("sector")["max_doy_raw"].std()
bl_amp_std = annual[mask_bl].groupby("sector")["amplitude_raw_yr"].std()

annual["max_doy_raw_anom_2015_z"]   = (annual["max_doy_raw_anom_2015"]
                                       / annual["sector"].map(bl_doy_std))
annual["amplitude_raw_anom_2015_z"] = (annual["amplitude_raw_anom_2015"]
                                       / annual["sector"].map(bl_amp_std))

# --- Shared legend builder -------------------------------------------------

def decade_legend_handles(smooth_color="#2C2C2A", smooth_lw=2.0):
    handles = [
        plt.scatter([], [], color=c, s=50,
                    edgecolors="white", linewidth=0.5, label=l)
        for c, l in DECADE_LEGEND
    ]
    if smooth_color:
        handles.append(
            Line2D([0], [0], color=smooth_color, lw=smooth_lw,
                   label="5-yr running mean")
        )
    return handles


# --- Panels 3a & 3b — timeseries small multiples --------------------------
# One subplot per sector. The 5-yr running mean is drawn in the sector colour
# so each panel feels self-contained.

def plot_anomaly_timeseries(var, ylabel, suptitle, outfile, is_days=True):
    plot_sectors = SECTORS_NO_CIRC + ["SIE_circumpolar"]
    fig, axes = plt.subplots(1, len(plot_sectors),
                             figsize=(len(plot_sectors) * 3.8, 5),
                             sharey=False, sharex=True)

    for ax, sec in zip(axes, plot_sectors):
        sub  = annual[annual["sector"] == sec].sort_values("Year")
        yrs  = sub["Year"].values
        vals = sub[var].values

        zero_line(ax)
        vline2016(ax)

        for yr, val in zip(yrs, vals):
            ax.scatter(yr, val, color=decade_color(yr),
                       s=50, zorder=4, edgecolors="white", linewidth=0.5)

        if len(vals) >= 5:
            smooth = uniform_filter1d(vals, size=5, mode="nearest")
            ax.plot(yrs, smooth, color=SECTOR_COLORS[sec],
                    lw=2.0, zorder=3, alpha=0.9)

        post_mean = sub[sub["Year"] >= 2016][var].mean()
        label_str = (f"Post-2016: {post_mean:+.1f} d" if is_days
                     else f"Post-2016: {post_mean:+.3f} Mkm²")
        ax.text(0.97, 0.04, label_str, transform=ax.transAxes,
                fontsize=8, color="#D4537E", ha="right",
                path_effects=stroke())

        ax.set_title(SECTOR_LABELS[sec], fontsize=11, fontweight="bold", pad=8)
        ax.tick_params(labelsize=9)
        ax.set_xlim(1977, yr_max + 1)

        if ax == axes[0]:
            ax.set_ylabel(ylabel, fontsize=11, labelpad=8)
        ax.set_xlabel("Year", fontsize=10, color="#5F5E5A")

        if sec == "SIE_circumpolar":
            ax.spines["left"].set_linewidth(2.0)
            ax.spines["left"].set_color("#B4B2A9")

    fig.legend(handles=decade_legend_handles(smooth_color=None),
               loc="lower center", ncol=7,
               fontsize=9, bbox_to_anchor=(0.5, -0.06), frameon=False)
    fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    save_fig(fig, outfile, OUTPUT_DIR)


print("Panel 3a: phase anomaly timeseries")
plot_anomaly_timeseries(
    var      = "max_doy_raw_anom_2015",
    ylabel   = "Phase anomaly (days)\n← Ahead of phase  |  Behind phase →",
    suptitle = "Timing of Sea Ice Maximum — Anomaly from 1979–2015 Baseline",
    outfile  = "fig_phase_amplitude_timeseries_3a_phase.png",
    is_days  = True,
)

print("Panel 3b: amplitude anomaly timeseries")
plot_anomaly_timeseries(
    var      = "amplitude_raw_anom_2015",
    ylabel   = "Amplitude anomaly (million km²)\n← Smaller  |  Larger →",
    suptitle = "Seasonal Amplitude — Anomaly from 1979–2015 Baseline",
    outfile  = "fig_phase_amplitude_timeseries_3b_amplitude.png",
    is_days  = False,
)


# --- Panel 3c — selected sectors side by side ------------------------------
# Weddell, Ross, King Haakon chosen as the three most dynamically distinct.
# Phase on top row, amplitude below, so you can read across a sector column
# and see how the two variables co-vary.

print("Panel 3c: selected sectors combined")

SELECTED = {
    "SIE_Weddell"    : "#2196F3",
    "SIE_Ross"       : "#4CAF50",
    "SIE_King_Haakon": "#9C27B0",
}

fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True, sharey="row")

row_vars    = ["max_doy_raw_anom_2015", "amplitude_raw_anom_2015"]
row_ylabels = [
    "Phase anomaly (days)\n← Ahead  |  Behind →",
    "Amplitude anomaly (million km²)\n← Smaller  |  Larger →",
]
row_titles = ["Phase", "Amplitude"]

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
                       s=50, zorder=4, edgecolors="white", linewidth=0.5)

        if len(vals) >= 5:
            smooth = uniform_filter1d(vals, size=5, mode="nearest")
            ax.plot(yrs, smooth, color=color, lw=2.5, zorder=3, alpha=0.9)

        post_mean = sub[sub["Year"] >= 2016][var].mean()
        is_days   = "doy" in var
        ax.text(0.97, 0.04,
                f"Post-2016: {post_mean:+.1f} d" if is_days
                else f"Post-2016: {post_mean:+.3f} Mkm²",
                transform=ax.transAxes, fontsize=9,
                color="#D4537E", ha="right", path_effects=stroke())

        if row == 0:
            ax.set_title(SECTOR_LABELS[sec], fontsize=13,
                         fontweight="bold", pad=10, color=color)
        if col == 0:
            ax.set_ylabel(ylabel, fontsize=10, labelpad=8)
        if col == 2:
            ax.text(1.03, 0.5, row_titles[row],
                    transform=ax.transAxes, fontsize=11,
                    fontweight="bold", color="#2C2C2A",
                    va="center", rotation=270)
        if row == 1:
            ax.set_xlabel("Year", fontsize=10, color="#5F5E5A")

        ax.tick_params(labelsize=10)
        ax.set_xlim(1977, yr_max + 1)

fig.legend(handles=decade_legend_handles(),
           loc="lower center", ncol=7,
           fontsize=9, bbox_to_anchor=(0.5, -0.04), frameon=False)
fig.suptitle(
    "Phase and Amplitude Anomaly — Selected Sectors\n"
    "Anomaly from 1979–2015 Baseline",
    fontsize=14, fontweight="bold", y=1.01
)
fig.tight_layout(rect=[0, 0.04, 0.97, 1])
save_fig(fig, "fig_phase_amplitude_timeseries_3c_selected.png", OUTPUT_DIR)


# --- Panel 3d — rolling variance -------------------------------------------
# This is where the post-2016 variance increase becomes visible. The fill
# makes it easy to see widening spread even across small multiples.

print("Panel 3d: rolling variance")

plot_sectors_roll = SECTORS_NO_CIRC + ["SIE_circumpolar"]
fig, axes = plt.subplots(2, len(plot_sectors_roll),
                         figsize=(len(plot_sectors_roll) * 3.2, 7),
                         sharey="row", sharex=True)

row_vars   = ["max_doy_raw_anom_2015", "amplitude_raw_anom_2015"]
row_labels = [
    "10-yr rolling std dev of phase (days)",
    "10-yr rolling std dev of amplitude (million km²)",
]

for row, (var, ylabel) in enumerate(zip(row_vars, row_labels)):
    for col, sec in enumerate(plot_sectors_roll):
        ax   = axes[row, col]
        sub  = (annual[annual["sector"] == sec]
                .sort_values("Year").set_index("Year"))
        roll = sub[var].rolling(10, center=True, min_periods=6).std()

        ax.plot(roll.index, roll.values,
                color=SECTOR_COLORS[sec], lw=2.0, zorder=3)
        ax.fill_between(roll.index, roll.values,
                        color=SECTOR_COLORS[sec], alpha=0.15, zorder=2)
        shade2016(ax, yr_max=yr_max)
        ax.set_xlim(yr_min + 4, yr_max - 4)

        if row == 0:
            ax.set_title(SECTOR_LABELS[sec], fontweight="bold", fontsize=10)
        if col == 0:
            ax.set_ylabel(ylabel, fontsize=9)
        if row == 1:
            ax.set_xlabel("Year", fontsize=9)
            ax.tick_params(axis="x", rotation=30)

fig.suptitle(
    "Has Variability Changed Over Time? — Phase and Amplitude\n"
    "10-year rolling standard deviation by sector",
    fontsize=13, fontweight="bold", y=1.01
)
fig.tight_layout()
save_fig(fig, "fig_phase_amplitude_timeseries_3d_rolling_variance.png", OUTPUT_DIR)


# --- Panel 3e — pre/post-2016 variability bars -----------------------------
# Simple summary of the rolling variance result above. Solid bars = pre-2016,
# hatched = post-2016. The short post-2016 window means these std devs are
# noisier — the footnote flags this so it doesn't get lost in review.

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

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

for ax, pre_vals, post_vals, ylabel, title, fmt in zip(
    axes,
    [phase_pre,  amp_pre],
    [phase_post, amp_post],
    ["Std dev of phase anomaly (days)",
     "Std dev of amplitude anomaly (million km²)"],
    ["Phase Variability\n(timing of maximum)",
     "Amplitude Variability\n(size of seasonal cycle)"],
    ["{:.1f}d", "{:.2f}"]
):
    bars_pre  = ax.bar(x - width/2, pre_vals,  width,
                       color=colors_bar, alpha=1.0,
                       edgecolor="white", label="1979–2015", zorder=3)
    bars_post = ax.bar(x + width/2, post_vals, width,
                       color=colors_bar, alpha=0.45,
                       edgecolor="white", label="2016–present",
                       hatch="///", zorder=3)

    for bar, val in zip(bars_pre, pre_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(pre_vals) * 0.02,
                fmt.format(val),
                ha="center", va="bottom", fontsize=9,
                fontweight="bold", path_effects=stroke())

    for bar, val in zip(bars_post, post_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(pre_vals) * 0.02,
                fmt.format(val),
                ha="center", va="bottom", fontsize=9,
                color="#5F5E5A", path_effects=stroke())

    ax.set_xticks(x)
    ax.set_xticklabels(labels_bar, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontweight="bold", fontsize=12)
    ax.set_ylim(0, max(max(pre_vals), max(post_vals)) * 1.35)
    ax.legend(fontsize=10, loc="upper right")

fig.text(
    0.5, -0.02,
    "Note: post-2016 period is shorter — std dev estimates are less stable",
    ha="center", fontsize=9, color="#5F5E5A", style="italic"
)
fig.suptitle("Has Variability Changed? Pre vs Post 2016",
             fontsize=14, fontweight="bold")
fig.tight_layout()
save_fig(fig, "fig_phase_amplitude_timeseries_3e_prepost_bars.png", OUTPUT_DIR)


# --- Panel 3f — overall phase and amplitude variability by sector ----------
# Three bars per sector: melt onset timing std dev, freeze onset timing std dev,
# amplitude std dev. This is the full-record variability picture — no pre/post
# split, just how much each sector varies overall. Useful context before the
# pre/post-2016 breakdown in panel 3e.

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

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

panels_f = [
    (axes[0], phase_std_max, "Melt onset timing\n(max DOY std dev)",   "Std dev (days)",          "{:.1f}d"),
    (axes[1], phase_std_min, "Freeze onset timing\n(min DOY std dev)", "Std dev (days)",          "{:.1f}d"),
    (axes[2], amp_std,       "Amplitude variability\n(max–min range)", "Std dev (million km²)",   "{:.2f}"),
]

for ax, vals, title, ylabel, fmt in panels_f:
    bars = ax.bar(x_f, vals, color=colors_f, width=0.6,
                  edgecolor="white", zorder=3)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vals) * 0.02,
                fmt.format(val),
                ha="center", va="bottom",
                fontsize=10, fontweight="bold", path_effects=stroke())
    ax.set_xticks(x_f)
    ax.set_xticklabels(labels_f, rotation=25, ha="right", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontweight="bold", fontsize=12)
    ax.set_ylim(0, max(vals) * 1.3)

fig.suptitle("Phase and Amplitude Variability by Sector  (1979–2023)",
             fontsize=14, fontweight="bold")
fig.tight_layout()
save_fig(fig, "fig_phase_amplitude_timeseries_3f_variability_by_sector.png", OUTPUT_DIR)


# --- Panel 3g — season length (tabled) ------------------------------------
# Kept here so the code isn't lost. Not included in the main figure sequence.
# Uncomment and run independently if needed for supplementary.

# print("Panel 3g: season length (tabled)")
# fig, axes = plt.subplots(2, len(SECTORS_NO_CIRC), figsize=(18, 7), sharey="row")
# ... see Plot_ch3_figures.py Fig 4 for the original implementation


print("\nAll fig_phase_amplitude_timeseries panels complete.")