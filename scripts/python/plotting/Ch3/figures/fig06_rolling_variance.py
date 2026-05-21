"""
fig_rolling_variance.py
=======================
10-year rolling standard deviation of phase and amplitude anomalies
by sector. Fig 6 in Chapter 3.

Two-panel figure:
  (a) Phase variability (days)
  (b) Amplitude variability (million km²)

Each panel shows all 5 sectors + circumpolar as lines.
Shaded grey region marks post-2016 period.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = "/user/geog/falejandraperez/sea-ice-phase/scripts/R/Ch3/data"
OUTPUT_DIR = "/user/geog/falejandraperez/sea-ice-phase/scripts/python/plotting/Ch3/figures"
GDRIVE     = "gdrive:My Drive/sea-ice-phase/results/Ch3_Figures"

ANNUAL_CSV = os.path.join(DATA_DIR, "annual_params.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family"      : "Nimbus Sans",
    "font.size"        : 10,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.linewidth"   : 0.8,
    "figure.dpi"       : 150,
    "savefig.dpi"      : 300,
    "savefig.bbox"     : "tight",
    "savefig.facecolor": "white",
})

SECTOR_COLORS = {
    "SIE_Weddell"                : "#2196F3",
    "SIE_Amundsen_Bellingshausen": "#F44336",
    "SIE_Ross"                   : "#4CAF50",
    "SIE_East_Antarctica"        : "#FF9800",
    "SIE_King_Haakon"            : "#9C27B0",
    "SIE_circumpolar"            : "#222222",
}
SECTOR_LABELS = {
    "SIE_Weddell"                : "Weddell",
    "SIE_Amundsen_Bellingshausen": "ABS",
    "SIE_Ross"                   : "Ross",
    "SIE_East_Antarctica"        : "East Antarctica",
    "SIE_King_Haakon"            : "King Haakon",
    "SIE_circumpolar"            : "Circumpolar",
}
SECTORS = list(SECTOR_COLORS.keys())

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading data...")
annual = pd.read_csv(ANNUAL_CSV)
for col in ["max_doy_raw", "amplitude_raw_yr",
            "max_doy_raw_anom", "amplitude_raw_anom"]:
    if col in annual.columns:
        annual[col] = pd.to_numeric(annual[col], errors="coerce")

yr_min = int(annual["Year"].min())
yr_max = int(annual["Year"].max())
print(f"  {len(annual)} rows | {yr_min}–{yr_max}")

# ── Baselines ─────────────────────────────────────────────────────────────────
mask = annual["Year"].between(1979, 2015)
bl_doy = annual[mask].groupby("sector")["max_doy_raw"].median()
bl_amp = annual[mask].groupby("sector")["amplitude_raw_yr"].median()
annual["max_doy_raw_anom_2015"]   = (annual["max_doy_raw"]
                                     - annual["sector"].map(bl_doy))
annual["amplitude_raw_anom_2015"] = (annual["amplitude_raw_yr"]
                                     - annual["sector"].map(bl_amp))

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
fig.subplots_adjust(hspace=0.12, top=0.92, bottom=0.10,
                    left=0.10, right=0.97)

row_vars   = ["max_doy_raw_anom_2015", "amplitude_raw_anom_2015"]
row_labels = ["Phase variability (days)",
              "Amplitude variability (million km²)"]
panel_letters = ["(a)", "(b)"]

for ax, var, ylabel, letter in zip(axes, row_vars, row_labels, panel_letters):

    for sec in SECTORS:
        sub  = (annual[annual["sector"] == sec]
                .sort_values("Year")
                .set_index("Year"))
        roll = sub[var].rolling(10, center=True, min_periods=6).std()

        color = SECTOR_COLORS[sec]
        lw    = 2.2 if sec == "SIE_circumpolar" else 1.5
        ls    = "--" if sec == "SIE_circumpolar" else "-"
        ax.plot(roll.index, roll.values,
                color=color, lw=lw, ls=ls, zorder=3,
                label=SECTOR_LABELS[sec])

    # Post-2016 shading
    ax.axvspan(2016, yr_max + 1, alpha=0.08, color="#888888", zorder=0)
    ax.axvline(2016, color="#D4537E", lw=0.9, ls=":", alpha=0.7, zorder=4)

    ax.set_ylim(bottom=0)
    ax.set_xlim(yr_min + 4, yr_max - 4)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.text(0.02, 0.97, letter, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top")
    ax.tick_params(labelsize=9)

axes[1].set_xlabel("Year", fontsize=10)

# Shared title
fig.suptitle("10-year rolling standard deviation of phase and amplitude anomalies",
             fontsize=10, fontweight="bold")

# Legend below both panels
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=6,
           fontsize=8.5, bbox_to_anchor=(0.5, 0.01), frameon=False)
fig.subplots_adjust(bottom=0.12)

# ── Save + rclone ─────────────────────────────────────────────────────────────
fpath = os.path.join(OUTPUT_DIR, "fig06_rolling_variance.png")
fig.savefig(fpath, dpi=300, bbox_inches="tight")
print(f"Saved → {fpath}")
plt.close()

ret = os.system(f'rclone copy "{fpath}" "{GDRIVE}"')
if ret == 0:
    print(f"Synced → {GDRIVE}")
else:
    print(f"WARNING: rclone failed (exit code {ret})")

print("Done.")