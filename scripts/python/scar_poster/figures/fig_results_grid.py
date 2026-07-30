"""
fig_results_grid.py

Poster Fig 2: the 20-cell results grid. Sector x season, coloured by the
interaction coefficient, hatched where significant after FDR. One row per
test so the null is visible at a glance rather than asserted in text.

Reads the CSVs already written by wind_divergence_coupling_test.py.
Run from the directory containing them (or edit PATHS).
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

# label: (csv, panel title)
PANELS = [
    ("wind_divergence_binary_test.csv",              "Net divergence — pre/post 2016"),
    ("wind_divergence_binary_test_div_positive.csv", "Lead-opening divergence — pre/post 2016"),
    ("wind_divergence_binary_test_div_negative.csv", "Convergence — pre/post 2016"),
    ("wind_divergence_oceanstate_test.csv",               "Net divergence — SST-conditioned"),
    ("wind_divergence_oceanstate_test_div_positive.csv",  "Lead-opening divergence — SST-conditioned"),
    ("wind_divergence_oceanstate_test_div_negative.csv",  "Convergence — SST-conditioned"),
]

OUT = "fig_results_grid.png"


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
    return coef, sig, df


avail = [(c, t) for c, t in PANELS if os.path.exists(c)]
if not avail:
    raise SystemExit("No result CSVs found in this directory.")

# common symmetric colour scale across panels so they are comparable
allvals = []
for c, _ in avail:
    g, _, _ = grid_from(c)
    allvals.append(g[np.isfinite(g)])
vmax = np.percentile(np.abs(np.concatenate(allvals)), 98)

ncol = 3
nrow = int(np.ceil(len(avail) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(4.1 * ncol, 3.1 * nrow))
axes = np.atleast_1d(axes).ravel()

for ax, (csv, title) in zip(axes, avail):
    coef, sig, df = grid_from(csv)
    im = ax.imshow(coef, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    for i in range(len(SECTORS)):
        for j in range(len(SEASONS)):
            if sig[i, j]:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                            edgecolor="k", lw=2.2))
                ax.text(j, i, "*", ha="center", va="center",
                        fontsize=15, fontweight="bold")

    n_sig = int(sig.sum())
    n_tot = int(np.isfinite(coef).sum())
    ax.set_xticks(range(len(SEASONS)), SEASONS, fontsize=9)
    ax.set_yticks(range(len(SECTORS)),
                  [SHORT[s] for s in SECTORS], fontsize=9)
    ax.set_title(f"{title}\n{n_sig}/{n_tot} significant (FDR q=0.05)",
                 fontsize=9.5)

for ax in axes[len(avail):]:
    ax.axis("off")

cb = fig.colorbar(im, ax=axes.tolist(), orientation="horizontal",
                  fraction=0.04, pad=0.07)
cb.set_label("interaction coefficient  (day$^{-1}$ per unit wind stress)\n"
             "positive = more divergence per unit wind after 2016 / at higher SST",
             fontsize=9)

fig.suptitle("Wind–divergence sensitivity did not shift across 2016",
             fontsize=13, y=0.99)
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"-> {OUT}")

# printed summary for the poster text / advisor meeting
print("\nPanel summary:")
for csv, title in avail:
    _, sig, df = grid_from(csv)
    print(f"  {title}: {int(sig.sum())}/{len(df)} significant")