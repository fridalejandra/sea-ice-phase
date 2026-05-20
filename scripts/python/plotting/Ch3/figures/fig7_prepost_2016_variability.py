"""
fig_prepost2016_variability.py
==============================
Pre vs post-2016 variability of phase and amplitude anomalies.
Grouped bar chart comparing std dev before and after 2016 for each sector.

Fig 7 in Chapter 3.

Two lettered panels:
  (a) Phase variability (days)
  (b) Amplitude variability (million km²)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
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

def stroke():
    return [pe.withStroke(linewidth=2, foreground="white")]

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading data...")
annual = pd.read_csv(ANNUAL_CSV)
for col in ["max_doy_raw", "amplitude_raw_yr"]:
    if col in annual.columns:
        annual[col] = pd.to_numeric(annual[col], errors="coerce")

# ── Baselines ─────────────────────────────────────────────────────────────────
mask   = annual["Year"].between(1979, 2015)
bl_doy = annual[mask].groupby("sector")["max_doy_raw"].median()
bl_amp = annual[mask].groupby("sector")["amplitude_raw_yr"].median()
annual["max_doy_raw_anom_2015"]   = (annual["max_doy_raw"]
                                     - annual["sector"].map(bl_doy))
annual["amplitude_raw_anom_2015"] = (annual["amplitude_raw_yr"]
                                     - annual["sector"].map(bl_amp))

# ── Pre / post split ──────────────────────────────────────────────────────────
pre  = annual[annual["Year"] <  2016]
post = annual[annual["Year"] >= 2016]

phase_pre  = [pre[pre["sector"]==s]["max_doy_raw_anom_2015"].dropna().std()
              for s in SECTORS]
phase_post = [post[post["sector"]==s]["max_doy_raw_anom_2015"].dropna().std()
              for s in SECTORS]
amp_pre    = [pre[pre["sector"]==s]["amplitude_raw_anom_2015"].dropna().std()
              for s in SECTORS]
amp_post   = [post[post["sector"]==s]["amplitude_raw_anom_2015"].dropna().std()
              for s in SECTORS]

labels = [SECTOR_LABELS[s] for s in SECTORS]
colors = [SECTOR_COLORS[s] for s in SECTORS]
x      = np.arange(len(SECTORS))
width  = 0.35

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.subplots_adjust(wspace=0.30, top=0.88, bottom=0.18,
                    left=0.07, right=0.97)

panels = [
    (axes[0], phase_pre, phase_post,
     "Std dev of phase anomaly (days)",
     "Phase variability\n(timing of maximum)",
     "{:.1f}", "(a)"),
    (axes[1], amp_pre, amp_post,
     "Std dev of amplitude anomaly (million km²)",
     "Amplitude variability\n(size of seasonal cycle)",
     "{:.2f}", "(b)"),
]

for ax, pre_vals, post_vals, ylabel, title, fmt, letter in panels:

    bars_pre  = ax.bar(x - width/2, pre_vals, width,
                       color=colors, alpha=1.0,
                       edgecolor="white", label="1979–2015", zorder=3)
    bars_post = ax.bar(x + width/2, post_vals, width,
                       color=colors, alpha=0.45,
                       edgecolor="white", label="2016–present",
                       hatch="///", zorder=3)

    for bar, val in zip(bars_pre, pre_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(pre_vals) * 0.02,
                fmt.format(val),
                ha="center", va="bottom", fontsize=8.5,
                fontweight="bold", path_effects=stroke())

    for bar, val in zip(bars_post, post_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(pre_vals) * 0.02,
                fmt.format(val),
                ha="center", va="bottom", fontsize=8.5,
                color="#5F5E5A", path_effects=stroke())

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontweight="bold", fontsize=10, pad=6)
    ax.set_ylim(0, max(max(pre_vals), max(post_vals)) * 1.35)
    ax.legend(fontsize=9, loc="upper right")

    # Panel letter
    ax.text(0.02, 0.97, letter, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top")

fig.text(0.5, 0.01,
         "Note: post-2016 period spans only 8 years — std dev estimates are less stable",
         ha="center", fontsize=8.5, color="#5F5E5A", style="italic")

# ── Save + rclone ─────────────────────────────────────────────────────────────
fpath = os.path.join(OUTPUT_DIR, "fig_prepost2016_variability.png")
fig.savefig(fpath, dpi=300, bbox_inches="tight")
print(f"Saved → {fpath}")
plt.close()

ret = os.system(f'rclone copy "{fpath}" "{GDRIVE}"')
if ret == 0:
    print(f"Synced → {GDRIVE}")
else:
    print(f"WARNING: rclone failed (exit code {ret})")

print("Done.")