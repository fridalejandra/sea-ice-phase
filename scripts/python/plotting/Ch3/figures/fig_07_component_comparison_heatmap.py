#!/usr/bin/env python3
"""
fig_07_component_comparison_heatmap.py -- Fig. 7: what the decomposition adds beyond the SIE anomaly.

Panel (a): for each of the seven sector x index pairs, the same index and
season correlated with four summaries of the sector's ice: the seasonal SIE
anomaly from the invariant cycle (the traditional anomaly), the amplitude,
the day of maximum, and the seasonal mean of the raw APAC anomaly.
Panel (b), if t37c exists: the sector's own seasonal-mean meridional wind
in the advance and retreat seasons against the same four.

Reads  results/ch3/tables/t37_component_comparison.csv
       results/ch3/tables/t37c_wind_comparison.csv (optional)
       (both from compute_fig07_component_comparison.py)
Writes results/ch3/figures/fig07_component_comparison_heatmap.png
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ch3_config import TABLES_DIR, OUTPUT_DIR
import ch3_style

INK = "0.35"
comp_path = os.path.join(TABLES_DIR, "t37_component_comparison.csv")
wind_path = os.path.join(TABLES_DIR, "t37c_wind_comparison.csv")
if not os.path.exists(comp_path):
    sys.exit(f"{comp_path} not found; run compute_fig07_component_comparison.py first")

cols = ["r_sie_anom", "r_amplitude", "r_max_doy", "r_raw_anom"]
pcols = ["p_sie_anom", "p_amplitude", "p_max_doy", "p_raw_anom"]
col_labels = ["SIE anomaly\n(seasonal mean)", "Amplitude", "Day of\nmaximum", "Raw anomaly\n(seasonal mean)"]

SEASON_NAME = {"annual": "annual", "DJF": "DJF", "MAM": "MAM", "JJA": "JJA", "SON": "SON",
               "ADV": "Mar–Aug", "RET": "Oct–Jan"}
INDEX_NAME = {"Nino34": "Niño3.4", "SAM": "SAM", "ASL": "ASL", "ZW3R": "ZW3", "ZW3G": "ZW3 (Goyal)",
              "v-wind": "v wind", "speed": "wind speed"}
SECTOR_NAME = {"ABS": "Amundsen-Bellingshausen"}
INK = "0.35"
VMAX = 0.8
bold = ch3_style.bold_font_properties(size=9)


def load(path):
    c = pd.read_csv(path)
    c["row_label"] = [
        f"{SECTOR_NAME.get(s, s)}  ·  {INDEX_NAME.get(b, b)}, {SEASON_NAME.get(se, se)}"
        for s, b, se in zip(c["sector"], c["index_base"], c["season"])
    ]
    mat = c.set_index("row_label")[cols].astype(float)
    pm = c.set_index("row_label")[pcols].astype(float)
    pm.columns = cols
    return mat, pm


def draw(ax, mat, pm, letter, show_cols=True):
    im = ax.imshow(mat.values, cmap="RdBu_r", vmin=-VMAX, vmax=VMAX, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            r = mat.values[i, j]; p = pm.values[i, j]
            if np.isnan(r):
                ax.text(j, i, "n/a", ha="center", va="center", color=INK, fontsize=8.5)
                continue
            sig = (not np.isnan(p)) and p < 0.05
            ax.text(j, i, f"{r:+.2f}", ha="center", va="center", fontsize=8.5,
                    color="white" if abs(r) > 0.45 else "0.15",
                    fontproperties=bold if sig else None)
    ax.set_xticks(range(len(cols)))
    if show_cols:
        ax.set_xticklabels(col_labels, fontsize=9, fontproperties=bold)
        ax.xaxis.tick_top()
    else:
        ax.set_xticklabels([])
    ax.set_yticks(range(len(mat)))
    ax.set_yticklabels(mat.index, fontsize=8.5)
    ax.tick_params(length=0, colors=INK)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(1, len(mat)):
        ax.axhline(i - 0.5, color="white", lw=1.5)
    for j in range(1, len(cols)):
        ax.axvline(j - 0.5, color="white", lw=1.5)
    ax.text(-0.02, 1.0, letter, transform=ax.transAxes, ha="right", va="bottom",
            color=INK, fontproperties=bold)
    return im


mat_a, pm_a = load(comp_path)
have_wind = os.path.exists(wind_path)
if have_wind:
    wc = pd.read_csv(wind_path)
    # panel (b): v wind only, advance and retreat seasons (speed stays in the CSV)
    wc = wc[(wc["index_base"] == "v-wind") & (wc["season"].isin(["ADV", "RET"]))]
    wc = wc[wc["r_amplitude"].notna()]   # circumpolar has no amplitude/timing scalars; keep the panel to the five sectors
    wc.to_csv(wind_path.replace(".csv", "_plotted.csv"), index=False)
    mat_b, pm_b = load(wind_path.replace(".csv", "_plotted.csv"))
    n_a, n_b = len(mat_a), len(mat_b)
    fig, (ax_a, ax_b) = plt.subplots(
        2, 1, figsize=(6.5, 0.42 * (n_a + n_b) + 1.6),
        gridspec_kw={"height_ratios": [n_a, n_b], "hspace": 0.12})
    im = draw(ax_a, mat_a, pm_a, "(a)")
    draw(ax_b, mat_b, pm_b, "(b)", show_cols=False)
    cbar = fig.colorbar(im, ax=[ax_a, ax_b], fraction=0.035, pad=0.03)
else:
    fig, ax_a = plt.subplots(figsize=(6.5, 0.5 * len(mat_a) + 1.2))
    im = draw(ax_a, mat_a, pm_a, "")
    cbar = fig.colorbar(im, ax=ax_a, fraction=0.04, pad=0.04)
    fig.tight_layout()

cbar.set_label("correlation", color=INK, fontsize=9)
cbar.ax.tick_params(labelsize=8, colors=INK, length=3)
cbar.outline.set_visible(False)
out = os.path.join(OUTPUT_DIR, "fig07_component_comparison_heatmap.png")
fig.savefig(out, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"wrote {out}")
print(mat_a.round(2).to_string())
if have_wind:
    print(mat_b.round(2).to_string())