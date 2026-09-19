"""
fig_monthly_lag_correlations.py
================================
Two-panel figure showing monthly correlations for key pairs:

Left panel  — Lag correlations (annual scalar ~ monthly index)
Right panel — Contemporaneous monthly correlations (monthly ice ~ monthly index)

One row per key index (SAM, Niño3.4, ASL, ZW3R).
X-axis = calendar month (Jan-Dec).
Y-axis = Pearson r.
Significant points (FDR p<0.05) shown as filled circles.
Grey band = p>0.05 threshold.
Advance (Mar-Aug) and Retreat (Oct-Jan) seasons shaded.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = "/user/geog/falejandraperez/sea-ice-phase/scripts/R/Ch3/data"
OUTPUT_DIR = "/user/geog/falejandraperez/sea-ice-phase/scripts/python/plotting/Ch3/figures"
GDRIVE     = "gdrive:sea-ice-phase/results/Ch3_Figures/"

LAG_CSV   = os.path.join(DATA_DIR, "lag_correlations.csv")
CONT_CSV  = os.path.join(DATA_DIR, "contemporaneous_correlations.csv")

# ── Style ─────────────────────────────────────────────────────────────────────
SECTOR_COLORS = {
    "Weddell"        : "#2196F3",
    "ABS"            : "#F44336",
    "Ross"           : "#4CAF50",
    "East Antarctica": "#FF9800",
    "King Haakon"    : "#9C27B0",
}
SECTOR_ORDER = ["Weddell", "ABS", "Ross", "East Antarctica", "King Haakon"]
INDICES      = ["SAM", "Nino34", "ASL", "ZW3R"]
INDEX_LABELS = {"SAM": "SAM", "Nino34": "Niño3.4", "ASL": "ASL", "ZW3R": "ZW3R"}
MONTHS       = list(range(1, 13))
MONTH_LABELS = ["J","F","M","A","M","J","J","A","S","O","N","D"]

# Significance threshold for n=44: r ≈ ±0.297 (p=0.05, two-tailed)
SIG_THRESHOLD = 0.297

# Shoulder season shading
ADV_MONTHS = [3, 4, 5, 6, 7, 8]    # Mar-Aug advance
RET_MONTHS = [10, 11, 12, 1]        # Oct-Jan retreat

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading data...")
lag  = pd.read_csv(LAG_CSV)
cont = pd.read_csv(CONT_CSV)

# Filter to amplitude only for main figure (cleaner story)
# Phase as separate figure
for var_label, var_col, outfile_suffix in [
    ("amplitude", "amplitude", "amplitude"),
    ("phase",     "phase",     "phase"),
    ("mean_sie",  "mean_sie",  "mean_sie"),
]:
    lag_sub  = lag[lag["var_type"]  == var_col] if var_col != "mean_sie" else pd.DataFrame()
    cont_sub = cont[cont["var_type"] == var_col]

    if len(cont_sub) == 0 and var_col != "phase":
        continue

    # ── Figure ────────────────────────────────────────────────────────────────
    n_indices = len(INDICES)
    n_cols = 1 if var_col == "phase" else 2
    fig, axes = plt.subplots(n_indices, n_cols, squeeze=False,
                             figsize=(13, 3.2 * n_indices),
                             sharey=False, sharex=True)

    fig.subplots_adjust(hspace=0.35, wspace=0.25,
                        left=0.08, right=0.97,
                        top=0.93, bottom=0.08)

    panel_letters = list("abcdefghijklmnopqrstuvwxyz")
    letter_idx    = 0

    for row, idx_name in enumerate(INDICES):

        col_defs = [(lag_sub[lag_sub["index"] == idx_name] if len(lag_sub) > 0 else pd.DataFrame(), "Lag correlation\n(annual scalar ~ monthly index)")] if var_col == "phase" else [(lag_sub[lag_sub["index"] == idx_name] if len(lag_sub) > 0 else pd.DataFrame(), "Lag correlation\n(annual scalar ~ monthly index)"), (cont_sub[cont_sub["index"] == idx_name], "Contemporaneous\n(monthly ice ~ monthly index)")]
        for col, (df_plot, col_title) in enumerate(col_defs):
            ax = axes[row, col]

            # Shoulder season shading
            for m in ADV_MONTHS:
                ax.axvspan(m - 0.5, m + 0.5, alpha=0.08,
                           color="#4CAF50", zorder=0)
            for m in RET_MONTHS:
                if m == 1:
                    ax.axvspan(0.5, 1.5, alpha=0.08, color="#F44336", zorder=0)
                else:
                    ax.axvspan(m - 0.5, m + 0.5, alpha=0.08,
                               color="#F44336", zorder=0)

            # Significance threshold band
            ax.axhspan(-SIG_THRESHOLD, SIG_THRESHOLD,
                       alpha=0.08, color="#888888", zorder=0)
            ax.axhline(0, color="#2C2C2A", lw=0.8, zorder=1)
            ax.axhline( SIG_THRESHOLD, color="#888888", lw=0.6,
                       ls="--", zorder=1, alpha=0.6)
            ax.axhline(-SIG_THRESHOLD, color="#888888", lw=0.6,
                       ls="--", zorder=1, alpha=0.6)

            if len(df_plot) > 0:
                for sec in SECTOR_ORDER:
                    sec_data = df_plot[df_plot["sector"] == sec].sort_values("month")
                    if len(sec_data) == 0:
                        continue

                    color = SECTOR_COLORS[sec]
                    x = sec_data["month"].values
                    y = sec_data["pearson_r"].values
                    s = sec_data["sig"].values

                    # Line
                    ax.plot(x, y, color=color, lw=1.5,
                            alpha=0.8, zorder=2)

                    # Significant points — filled
                    ax.scatter(x[s],  y[s],
                               color=color, s=45, zorder=4,
                               edgecolors="white", linewidth=0.5)

                    # Non-significant — open
                    ax.scatter(x[~s], y[~s],
                               color=color, s=25, zorder=3,
                               facecolors="none",
                               edgecolors=color, linewidth=0.8,
                               alpha=0.5)

            ax.set_xlim(0.5, 12.5)
            ax.set_xticks(MONTHS)
            ax.set_xticklabels(MONTH_LABELS, fontsize=9)
            ax.tick_params(axis="y", labelsize=9)
            ax.spines[["top","right"]].set_visible(False)

            # Panel letter
            ax.text(0.02, 0.97,
                    f"({panel_letters[letter_idx]})",
                    transform=ax.transAxes,
                    fontsize=9, fontweight="bold", va="top")
            letter_idx += 1

            # Index label on left column
            if col == 0:
                ax.set_ylabel(f"{INDEX_LABELS[idx_name]}\nPearson r",
                              fontsize=9)

            # Column titles on top row
            if row == 0:
                ax.set_title(col_title, fontsize=10,
                             fontweight="bold", pad=6)

            # X label on bottom row
            if row == n_indices - 1:
                ax.set_xlabel("Month", fontsize=9)

    # ── Legend ────────────────────────────────────────────────────────────────
    sector_handles = [
        Line2D([0], [0], color=SECTOR_COLORS[s], lw=2, label=s)
        for s in SECTOR_ORDER
    ]
    season_handles = [
        mpatches.Patch(facecolor="#4CAF50", alpha=0.3, label="Advance (Mar–Aug)"),
        mpatches.Patch(facecolor="#F44336", alpha=0.3, label="Retreat (Oct–Jan)"),
        mpatches.Patch(facecolor="#888888", alpha=0.2, label="p>0.05 band"),
    ]
    all_handles = sector_handles + season_handles
    fig.legend(handles=all_handles, loc="lower center", ncol=8,
               fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.5, 0.01))
    fig.subplots_adjust(bottom=0.08)

    # ── Save ──────────────────────────────────────────────────────────────────
    outfile = os.path.join(OUTPUT_DIR,
                           f"fig11_monthly_lag_correlations_{outfile_suffix}.png")
    fig.savefig(outfile, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved → {outfile}")
    plt.close()

    ret = os.system(f'rclone copy "{outfile}" "{GDRIVE}"')
    if ret == 0:
        print(f"Synced → {GDRIVE}")

print("\nDone.")