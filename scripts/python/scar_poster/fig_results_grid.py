"""
fig_results_grid.py

Poster figure: 6-panel results grid. Sector × season, coloured by
interaction coefficient, starred where significant after FDR.
Clean version — no main title, no significance counts in subtitles.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SECTORS = ["Amundsen-Bellingshausen", "Weddell", "King Haakon VII",
           "East Antarctica", "Ross-Amundsen"]
SHORT = {"Amundsen-Bellingshausen": "ABS", "Weddell": "WED",
         "King Haakon VII": "KHV", "East Antarctica": "EA",
         "Ross-Amundsen": "RA"}
SEASONS = ["DJF", "MAM", "JJA", "SON"]

PANELS = [
    ("wind_divergence_binary_test.csv",              "Net"),
    ("wind_divergence_binary_test_div_positive.csv", "Lead-opening"),
    ("wind_divergence_binary_test_div_negative.csv", "Convergence"),
    ("wind_divergence_oceanstate_test.csv",               "Net"),
    ("wind_divergence_oceanstate_test_div_positive.csv",  "Lead-opening"),
    ("wind_divergence_oceanstate_test_div_negative.csv",  "Convergence"),
]
ROW_LABELS = ["Pre/post 2016", "SST-conditioned"]

OUT = "fig_results_grid.png"
RCLONE_REMOTE = "gdrive:scar_poster/"


def grid_from(csv):
    df = pd.read_csv(csv)
    coef = np.full((len(SECTORS), len(SEASONS)), np.nan)
    sig = np.zeros_like(coef, dtype=bool)
    for i, s in enumerate(SECTORS):
        for j, sea in enumerate(SEASONS):
            r = df[(df.sector == s) & (df.season == sea)]
            if len(r) == 0:
                continue
            coef[i, j] = r.iloc[0]["interaction_coef"]
            sig[i, j] = bool(r.iloc[0].get("significant_fdr", False))
    return coef, sig


avail = [(c, t) for c, t in PANELS if os.path.exists(c)]
if not avail:
    raise SystemExit("No result CSVs found in this directory.")

allvals = []
for c, _ in avail:
    g, _ = grid_from(c)
    allvals.append(g[np.isfinite(g)])
vmax = np.percentile(np.abs(np.concatenate(allvals)), 98)

nrow = 2
ncol = 3
fig, axes = plt.subplots(nrow, ncol, figsize=(11, 6.5))

for idx, (ax, (csv, title)) in enumerate(zip(axes.flat, avail)):
    coef, sig = grid_from(csv)
    im = ax.imshow(coef, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    for i in range(len(SECTORS)):
        for j in range(len(SEASONS)):
            if sig[i, j]:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                            edgecolor="k", lw=2.2))
                ax.text(j, i, "*", ha="center", va="center",
                        fontsize=16, fontweight="bold")

    ax.set_xticks(range(len(SEASONS)))
    ax.set_xticklabels(SEASONS if idx >= ncol else [], fontsize=11)
    ax.set_yticks(range(len(SECTORS)))
    ax.set_yticklabels([SHORT[s] for s in SECTORS] if idx % ncol == 0 else [],
                       fontsize=11)

    if idx < ncol:
        ax.set_title(title, fontsize=13, fontweight="bold")

for row, label in enumerate(ROW_LABELS):
    axes[row, 0].text(-0.35, 0.5, label, transform=axes[row, 0].transAxes,
                      fontsize=12, fontweight="bold", va="center", ha="right",
                      rotation=90)

fig.subplots_adjust(hspace=0.15, wspace=0.08)

cb = fig.colorbar(im, ax=axes.ravel().tolist(), orientation="vertical",
                  fraction=0.025, pad=0.02)
cb.set_label("interaction coefficient (day⁻¹ per unit wind stress)",
             fontsize=11)

fig.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"-> {OUT}")

os.system(f"rclone copy {OUT} {RCLONE_REMOTE}")
print("uploaded.")

for csv, title in avail:
    _, sig = grid_from(csv)
    print(f"  {title}: {int(sig.sum())}/20 significant")