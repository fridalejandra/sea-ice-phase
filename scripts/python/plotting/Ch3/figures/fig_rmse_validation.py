"""
How much does each model component actually help?

Plots the sequential RMSE improvement over the invariant seasonal cycle
as we add amplitude, then phase, then both together. One grouped bar cluster
per sector plus circumpolar. Negative bars mean that model is worse than
doing nothing.

Needs rmse_summary.csv from the processing scripts. This is computed in R where the bulk stats/ GAM analysis is done. 
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ch3_style import (
    apply_style,
    SECTORS_NO_CIRC, SECTOR_LABELS,
    stroke, save_fig, DEFAULT_OUTPUT_DIR,
)

apply_style()

DATA_DIR   = "/user/geog/falejandraperez/sea-ice-phase/scripts/R/chapter3/data"
OUTPUT_DIR = DEFAULT_OUTPUT_DIR
RMSE_CSV   = os.path.join(DATA_DIR, "rmse_summary.csv")

# --- Load ------------------------------------------------------------------

print("Loading RMSE data...")
rmse = pd.read_csv(RMSE_CSV)

# Make sure sectors appear in the same left-to-right order as every other
# Ch3 figure, with circumpolar tacked on at the end.
all_sectors = SECTORS_NO_CIRC + ["SIE_circumpolar"]

sector_order = pd.DataFrame({
    "sector": all_sectors,
    "label" : [SECTOR_LABELS[s] for s in all_sectors],
})
rmse_ordered = sector_order.merge(rmse, on="sector", how="left")

# --- Model definitions -----------------------------------------------------
# Colours reflect the cumulative logic: grey = baseline, then we layer in
# amplitude (blue), phase (pink), and finally both together (green).

MODELS = [
    ("pct_imp_iac",   "Invariant",   "#B4B2A9"),
    ("pct_imp_amp",   "Amplitude",   "#378ADD"),
    ("pct_imp_phase", "Phase",       "#D4537E"),
    ("pct_imp_apac",  "Amp + Phase", "#1D9E75"),
]

# --- Plot ------------------------------------------------------------------

n_sectors = len(all_sectors)
n_models  = len(MODELS)
x         = np.arange(n_sectors)
width     = 0.18
gap       = (n_models - 1) / 2

fig, ax = plt.subplots(figsize=(13, 6))

for i, (col, label, color) in enumerate(MODELS):
    vals = rmse_ordered[col].tolist()
    xpos = x + (i - gap) * width

    bars = ax.bar(xpos, vals, width,
                  color=color, edgecolor="white",
                  label=label, zorder=3)

    for bar, val in zip(bars, vals):
        if pd.notna(val):
            ypos = (bar.get_height() + 0.5
                    if val >= 0 else bar.get_height() - 2.5)
            ax.text(
                bar.get_x() + bar.get_width() / 2, ypos,
                f"{val:.0f}%",
                ha="center", va="bottom",
                fontsize=7.5, color=color, fontweight="bold",
                path_effects=stroke()
            )

ax.axhline(0, color="grey", lw=0.7, ls="--", zorder=0)

# Light separator before the circumpolar cluster so it reads as distinct
ax.axvline(n_sectors - 1 - 0.5, color="#B4B2A9",
           lw=0.8, ls=":", zorder=1, alpha=0.7)
ax.text(n_sectors - 1, ax.get_ylim()[0], "circumpolar",
        ha="center", va="bottom", fontsize=8,
        color="#B4B2A9", style="italic")

ax.set_xticks(x)
ax.set_xticklabels(
    [SECTOR_LABELS[s] for s in all_sectors],
    rotation=20, ha="right", fontsize=11
)
ax.set_ylabel("RMSE improvement over invariant seasonal cycle (%)", fontsize=11)
ax.set_title(
    "Sequential RMSE Improvement by Sector\n"
    "Each model adds one component above the previous",
    fontweight="bold"
)

# Scale to the data so negative bars and annotations are never clipped
all_vals = rmse_ordered[["pct_imp_iac", "pct_imp_amp",
                          "pct_imp_phase", "pct_imp_apac"]].values.flatten()
ax.set_ylim(bottom=min(0, np.nanmin(all_vals)) * 1.4,
            top=np.nanmax(all_vals) * 1.25)

ax.legend(title="Model", fontsize=10, loc="upper left", title_fontsize=10)

fig.tight_layout()
save_fig(fig, "fig_rmse_validation.png", OUTPUT_DIR)