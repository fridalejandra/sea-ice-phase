#!/usr/bin/env python3
"""
fig_07_abs_growth_season.py — Figure 7: Amundsen-Bellingshausen amplitude
vs. growth-season length (day of max minus day of min), 2016-2023 (§3.2).
Compare the plotted r to the pre-2016 r=+0.09 reported in the text.

Split out of build_ch3_figures.py (2026-09-18) so each manuscript figure has
its own script, named to match: fig_##_name.py. Figure logic unchanged from
that file's Fig-7 block, INCLUDING the period-filter fix applied 2026-09-18:
annual_params.csv carries two `period` values per sector-year for 1979-2018
(FULL and HR2018, the truncated reproduction window). Unfiltered, 2016-2018
each appeared twice (n=11 instead of 8), inflating apparent significance
(r=0.827, p=0.0017 unfiltered vs. the correct r=0.812, p=0.014) -- verified
against the real annual_params.csv before this fix was applied. Filtering to
period=="FULL" is not optional here.

Run this from the same directory as ch3_config.py
(scripts/python/plotting/Ch3/figures/), after 01_fit_apac.R has produced
annual_params.csv (no further pipeline steps needed for this one figure).

Reads:
    data/ch3/annual_params.csv
Writes:
    results/ch3/figures/fig7_abs_growth_season.png
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ch3_config import ANNUAL_CSV, SECTORS, OUTPUT_DIR

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def pick_col(df, candidates, label):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"Could not find a column for '{label}'. Tried: {candidates}\n"
        f"Actual columns in this table: {list(df.columns)}"
    )


ann = pd.read_csv(ANNUAL_CSV)
if "period" in ann.columns:
    ann = ann[ann["period"] == "FULL"]
abs_sector = [s for s in SECTORS if "Amundsen" in s][0]
year_col = pick_col(ann, ["Year", "year"], "year")
sector_col = pick_col(ann, ["sector", "Sector"], "sector")

ann_abs = ann[(ann[sector_col] == abs_sector) & (ann[year_col].between(2016, 2023))].sort_values(year_col)
if len(ann_abs) == 0:
    raise ValueError(
        f"No rows matched sector=={abs_sector!r} in {ANNUAL_CSV}. "
        f"Actual sector values present: {sorted(ann[sector_col].unique())}"
    )

amp_col = pick_col(ann_abs, ["amplitude_raw_yr", "amplitude_raw", "amplitude", "amplitude_raw_anom"], "ABS amplitude")
max_col = pick_col(ann_abs, ["max_doy_raw", "max_doy", "max_doy_raw_anom"], "ABS day of max")
min_col = pick_col(ann_abs, ["min_doy_raw", "min_doy", "min_doy_raw_anom"], "ABS day of min")
using_anom = amp_col.endswith("_anom")

years = ann_abs[year_col].tolist()
growth_len = (ann_abs[max_col] - ann_abs[min_col]).tolist()
amp = ann_abs[amp_col].tolist()
r, p = pearsonr(growth_len, amp)

fig, ax = plt.subplots(figsize=(5.6, 4.6))
sc = ax.scatter(growth_len, amp, c=years, cmap="viridis", s=90, edgecolor="k", linewidth=0.6, zorder=3)
for x, yv, yr in zip(growth_len, amp, years):
    ax.annotate(str(int(yr)), (x, yv), textcoords="offset points", xytext=(6, 4), fontsize=8)
m, b = np.polyfit(growth_len, amp, 1)
xx = np.linspace(min(growth_len) - 5, max(growth_len) + 5, 50)
ax.plot(xx, m * xx + b, color="#C0392B", lw=1.6, ls="--", zorder=2,
        label=f"r = {r:+.2f} (2016-2023, this run, p={p:.3f})")
ax.set_xlabel("Growth-season length, day of max - day of min (days)"
              + (" [anomaly]" if using_anom else ""))
ax.set_ylabel("Amplitude" + (" anomaly" if using_anom else "") + " (10⁶ km²)")
ax.set_title("Amundsen-Bellingshausen: amplitude vs. growth-season length\n"
              "(2016-2023; compare to the pre-2016 r=+0.09 reported in the text)", fontsize=9.5)
ax.legend(fontsize=8.5, frameon=False, loc="upper left")
fig.colorbar(sc, ax=ax, label="Year")
fig.tight_layout()
out7 = os.path.join(OUTPUT_DIR, "fig7_abs_growth_season.png")
fig.savefig(out7, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"wrote {out7}")
print(f"  r={r:.3f}, p={p:.4f}, using {'ANOMALY (raw columns not found)' if using_anom else 'raw'} columns")
