#!/usr/bin/env python3
"""
fig_10_atmosphere_sevenpairs.py — Figure 10: the seven pre-specified
atmosphere-component relationships, full-record correlation, Bonferroni
over 7 (§3.3).

Split out of build_ch3_figures.py (2026-09-18) so each manuscript figure has
its own script, named to match: fig_##_name.py. Figure logic unchanged from
that file's Fig-10 block.

Run this from the same directory as ch3_config.py
(scripts/python/plotting/Ch3/figures/), after ch3_stats.py has produced
t35_primary_pairs.csv.

Reads:
    results/ch3/tables/t35_primary_pairs.csv
Writes:
    results/ch3/figures/fig08_atmosphere_sevenpairs.png
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ch3_config import TABLES_DIR, OUTPUT_DIR
import ch3_style  # sets font (Helvetica/Tacoma) + spines for every figure


def load_table(name):
    path = os.path.join(TABLES_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path} — run the canonical pipeline first.")
    return pd.read_csv(path)


pairs = load_table("t35_primary_pairs.csv")

TARGET_LABEL = {"amplitude_raw_anom": "amplitude", "max_doy_raw_anom": "max-day"}


def family_of(index_col):
    for prefix, fam in [("Nino34", "ENSO"), ("SAM", "SAM"), ("ASL", "ASL"), ("ZW3", "ZW3")]:
        if index_col.startswith(prefix):
            return fam
    return "?"


pairs = pairs.copy()
pairs["label"] = pairs["sector"] + " · " + pairs["target"].map(TARGET_LABEL) + " ~ " + pairs["index"]
pairs["family"] = pairs["index"].map(family_of)
pairs = pairs.reindex(pairs["r"].abs().sort_values().index)

fig, ax = plt.subplots(figsize=(8, 4.8))
fam_colors = {"ENSO": "#1f77b4", "SAM": "#d62728", "ASL": "#2ca02c", "ZW3": "#9467bd"}
y = np.arange(len(pairs))[::-1]
for i, (_, row) in enumerate(pairs.iterrows()):
    ax.barh(y[i], row["r"], color=fam_colors.get(row["family"], "#888"), height=0.6)
    ax.text(row["r"] + (0.015 if row["r"] >= 0 else -0.015), y[i],
            f"r={row['r']:+.2f}, p_Bonf={row['p_bonf7']:.3f}",
            va="center", ha="left" if row["r"] >= 0 else "right", fontsize=8)
ax.axvline(0, color="k", lw=0.8)
ax.set_yticks(y); ax.set_yticklabels(pairs["label"], fontsize=9)
lim = max(0.75, pairs["r"].abs().max() + 0.2)
ax.set_xlim(-lim, lim)
ax.set_xlabel("Full-record correlation (1979-2023, detrended)")
ax.set_title("Seven pre-specified atmosphere-component relationships\n"
              "(Bonferroni over 7; all significant)", fontsize=9.5)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in fam_colors.values()]
ax.legend(handles, fam_colors.keys(), loc="lower right", frameon=False, fontsize=8.5, ncol=4)
fig.tight_layout()
out10 = os.path.join(OUTPUT_DIR, "fig08_atmosphere_sevenpairs.png")
fig.savefig(out10, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"wrote {out10}")
for _, row in pairs.iterrows():
    print(f"  {row['label']}: r={row['r']:+.3f}, p_bonf7={row['p_bonf7']:.4f}, LOO worst p={row['loo_worst_p']:.4f}")