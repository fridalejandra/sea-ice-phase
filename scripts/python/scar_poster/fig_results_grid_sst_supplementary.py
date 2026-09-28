"""
fig_results_grid_sst_supplementary.py

SUPPLEMENTARY figure: the SST-conditioned row that was dropped from the
main poster results grid. Same 3 panels (Net / Divergence / Convergence),
but the interaction term is wind x SST instead of wind x post.
Uploaded to gdrive:scar_poster/supplementary/.
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
    ("wind_divergence_oceanstate_test.csv",              "Net"),
    ("wind_divergence_oceanstate_test_div_positive.csv", "Divergence"),
    ("wind_divergence_oceanstate_test_div_negative.csv", "Convergence"),
]

OUT = "fig_results_grid_sst_conditioned_supplementary.png"
RCLONE_REMOTE = "gdrive:scar_poster/supplementary/"


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
    raise SystemExit("No SST-conditioned result CSVs found in this directory.")

allvals = []
for c, _ in avail:
    g, _ = grid_from(c)
    allvals.append(g[np.isfinite(g)])
vmax = np.percentile(np.abs(np.concatenate(allvals)), 98)

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

for idx, (ax, (csv, title)) in enumerate(zip(axes, avail)):
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
    ax.set_xticklabels(SEASONS, fontsize=11)
    ax.set_yticks(range(len(SECTORS)))
    ax.set_yticklabels([SHORT[s] for s in SECTORS] if idx == 0 else [], fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold")

fig.suptitle("SST-conditioned (supplementary)", fontsize=13, y=1.03)
fig.subplots_adjust(hspace=0.15, wspace=0.08)

cb = fig.colorbar(im, ax=axes.ravel().tolist(), orientation="vertical",
                  fraction=0.025, pad=0.02)
cb.set_label("interaction coefficient (day⁻¹ per unit SST anomaly)", fontsize=11)

fig.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"-> {OUT}")

os.system(f"rclone mkdir {RCLONE_REMOTE}")
os.system(f"rclone copy {OUT} {RCLONE_REMOTE}")
print("uploaded to supplementary.")

for csv, title in avail:
    _, sig = grid_from(csv)
    print(f"  {title}: {int(sig.sum())}/20 significant")