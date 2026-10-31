#!/usr/bin/env python
"""ch3_fig06_sst_components.py -- Fig. 3.6: detrended correlation, 1979-2024, between
HadISST SST (55-65 S, each sector) and (a) the annual mean, (b) the amplitude and
(c) the day of maximum, by SST season.
Reads results/ch3/tables/ch3_sst_partition_full.csv (from Ch3/ch3_sst_partition.py)."""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ch3_style as S
from ch3_config import TABLES_DIR

SRC = os.path.join(TABLES_DIR, "ch3_sst_partition_full.csv")
if not os.path.exists(SRC):
    sys.exit(f"{SRC} not found; run Ch3/ch3_sst_partition.py first")
full = pd.read_csv(SRC).set_index(["sector", "component", "season"])

SECS = [("WS", "Weddell"), ("KH", "King Haakon VII"), ("EA", "East Antarctica"),
        ("RA", "Ross"), ("ABS", "Amundsen–\nBellingshausen")]
SEAS = [("annual", "Annual"), ("DJF", "DJF"), ("MAM", "MAM"), ("JJA", "JJA"),
        ("SON", "SON"), ("advance", "Adv."), ("retreat", "Ret.")]
COMPS = [("annual mean", "Annual mean"), ("amplitude", "Amplitude"), ("day of max", "Day of maximum")]
CMAP = sns.color_palette("vlag", as_cmap=True)

fig, axes = plt.subplots(1, 3, figsize=(S.TEXT_WIDTH_IN, 2.7), sharey=True,
                         gridspec_kw=dict(wspace=0.08))
for ax, (comp, title) in zip(axes, COMPS):
    M = np.array([[full.loc[(c, comp, s), "r"] for s, _ in SEAS] for c, _ in SECS])
    im = ax.imshow(M, cmap=CMAP, vmin=-0.8, vmax=0.8, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]) and abs(M[i, j]) >= 0.3:
                ax.text(j, i, f"{M[i, j]:.2f}".replace("0.", ".").replace("-", "\u2212"), ha="center",
                        va="center", fontsize=5, color="white" if abs(M[i, j]) > 0.5 else S.INK)
    ax.set_title(title)
    ax.set_xticks(range(len(SEAS))); ax.set_xticklabels([l for _, l in SEAS], rotation=45, ha="right")
    ax.set_yticks(range(len(SECS))); ax.set_yticklabels([l for _, l in SECS])
    ax.tick_params(length=0, labelsize=7)
    for sp in ax.spines.values():
        sp.set_visible(False)
axes[1].set_xlabel("SST season")
cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02, ticks=[-0.8, -0.4, 0, 0.4, 0.8])
cb.ax.tick_params(labelsize=7)
cb.set_label("Detrended r", color=S.INK)
fig.text(0.5, -0.02, "Values shown where |r| \u2265 0.3", ha="center", fontsize=6.5, color=S.GREY) if False else None
cb.outline.set_visible(False)
S.label_panels(axes)
S.save_figure(fig, "ch3_fig06_sst_components")
