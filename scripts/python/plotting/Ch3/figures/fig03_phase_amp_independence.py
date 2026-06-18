"""
fig03_phase_amp_independence.py

Rolling 10-year Spearman correlation between phase and amplitude anomalies
by sector. Shows whether the two components carry independent information
and whether that independence has changed post-2016.

Figure 3 in Chapter 3.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, linregress

from ch3_style import (
    apply_style,
    SECTORS, SECTOR_COLORS, SECTOR_LABELS,
    vline2016, stroke,
    save_fig, DEFAULT_OUTPUT_DIR,
)

apply_style()

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR   = "/user/geog/falejandraperez/sea-ice-phase/scripts/R/Ch3/data"
OUTPUT_DIR = DEFAULT_OUTPUT_DIR
ANNUAL_CSV = os.path.join(DATA_DIR, "annual_params.csv")
WINDOW     = 10
CUTOFF     = 2016
DETREND    = True

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading annual params...")
params = pd.read_csv(ANNUAL_CSV)
for col in ["max_doy_raw_anom", "amplitude_raw_anom"]:
    params[col] = pd.to_numeric(params[col], errors="coerce")
print(f"  {len(params)} rows | {params['Year'].min()}–{params['Year'].max()}")


def detrend_series(years, values):
    mask = ~np.isnan(values)
    if mask.sum() < 5:
        return values
    slope, intercept, *_ = linregress(years[mask].astype(float),
                                      values[mask])
    return values - (slope * years.astype(float) + intercept)


# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(10, 6))
fig.suptitle(
    "Rolling phase–amplitude coupling by sector (1979–2023)",
    fontsize=11, fontweight="bold", y=1.01
)

LETTERS = list("abcdef")

for i, sec in enumerate(SECTORS):
    ax    = axes.flatten()[i]
    color = SECTOR_COLORS[sec]
    label = SECTOR_LABELS[sec]

    df = (params[params["sector"] == sec]
          .sort_values("Year")
          .dropna(subset=["max_doy_raw_anom", "amplitude_raw_anom"])
          .reset_index(drop=True)
          .copy())

    if DETREND:
        df["max_doy_raw_anom"]    = detrend_series(df["Year"].values,
                                                    df["max_doy_raw_anom"].values)
        df["amplitude_raw_anom"]  = detrend_series(df["Year"].values,
                                                    df["amplitude_raw_anom"].values)

    # Rolling Spearman r
    roll_r, years = [], []
    for j in range(WINDOW, len(df) + 1):
        w = df.iloc[j - WINDOW:j]
        r, _ = spearmanr(w["max_doy_raw_anom"], w["amplitude_raw_anom"])
        roll_r.append(r)
        years.append(df.iloc[j - 1]["Year"])

    years   = np.array(years)
    roll_r  = np.array(roll_r)

    # Pre/post means
    pre_mean  = np.nanmean(roll_r[years <= CUTOFF])
    post_mean = np.nanmean(roll_r[years >  CUTOFF])

    # Shaded independence band
    ax.fill_between(years, -0.4, 0.4, color="gray", alpha=0.07, zorder=1)

    # Reference lines
    ax.axhline(0,    color="k",    linewidth=0.8, alpha=0.5, zorder=2)
    ax.axhline(0.4,  color="gray", linewidth=0.6, linestyle=":", alpha=0.6, zorder=2)
    ax.axhline(-0.4, color="gray", linewidth=0.6, linestyle=":", alpha=0.6, zorder=2)

    # 2016 cutoff
    vline2016(ax, label="2016")

    # Rolling r line
    ax.plot(years, roll_r, color=color, linewidth=2,
            marker="o", markersize=3, zorder=4)

    # Pre/post mean annotations
    ax.text(0.04, 0.92, f"pre: {pre_mean:+.2f}",
            transform=ax.transAxes, fontsize=7.5,
            color="gray", fontweight="bold", va="top")
    ax.text(0.04, 0.82, f"post: {post_mean:+.2f}",
            transform=ax.transAxes, fontsize=7.5,
            color="#D4537E", fontweight="bold", va="top")

    # Panel letter + sector label
    ax.text(0.04, 0.06, f"({LETTERS[i]})",
            transform=ax.transAxes, fontsize=8,
            fontweight="bold", va="bottom", color="#2C2C2A")

    ax.set_title(label, fontsize=10, fontweight="bold", color=color, pad=4)
    ax.set_ylabel("Spearman r", fontsize=8)
    ax.set_ylim(-1, 1)
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if i >= 3:
        ax.set_xlabel("Year (end of 10-year window)", fontsize=8)

fig.tight_layout()
save_fig(fig, "fig03_phase_amp_independence.png", OUTPUT_DIR)
print("Saved fig03_phase_amp_independence.png")