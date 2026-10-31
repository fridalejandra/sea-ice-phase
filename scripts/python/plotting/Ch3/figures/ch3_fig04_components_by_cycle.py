#!/usr/bin/env python3
"""ch3_fig04_components_by_cycle.py -- Fig. 3.4: each cycle's extent anomaly and its
Trend, Amplitude and Phase components (cycle means, 21 Feb to 20 Feb), one row per
sector. Positive bars vermillion, negative blue; the two short lines in each panel give
the mean size (mean |bar|) over 1979-2015 and 2016-2024. No panel letters (house rule).

Reads  results/ch3/tables/t32_attribution_by_cycle.csv (rebuilt by make_ch3_figures.py
       from fig_03-04_attribution_annual.py before this runs; refuses a table older than
       data/ch3/daily_fitted.csv)
Writes results/ch3/tables/t32_component_size_era.csv
Landscape width (9 in), approved for this figure only."""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ch3_style as S
from ch3_config import SECTORS, TABLES_DIR, BREAK_YEAR, DAILY_CSV

SRC = os.path.join(TABLES_DIR, "t32_attribution_by_cycle.csv")
if not os.path.exists(SRC):
    sys.exit(f"{SRC} not found; run fig_03-04_attribution_annual.py first")
if os.path.getmtime(SRC) < os.path.getmtime(DAILY_CSV):
    sys.exit(f"{SRC} is older than {DAILY_CSV}: rebuild it with fig_03-04_attribution_annual.py")
t = pd.read_csv(SRC)
YR = "Year" if "Year" in t.columns else "cycle"

COLS = [("anomaly_from_iac", "Extent anomaly"), ("trend_component", "Trend"),
        ("amplitude_component", "Amplitude"), ("phase_component", "Phase"),
        ("residual_apac", "Raw anomaly")]
PLOT_SECTORS = [s for s in SECTORS if s != "SIE_circumpolar"]
PLOT_COLS = [(c, lab) for c, lab in COLS if c != "residual_apac"]
ROWLAB = {"SIE_Weddell": "Weddell", "SIE_Amundsen_Bellingshausen": "Amundsen–\nBellingshausen",
          "SIE_Ross": "Ross", "SIE_East_Antarctica": "East\nAntarctica", "SIE_King_Haakon": "King\nHaakon VII"}

# era sizes for every sector and component (table)
rows = []
for s in SECTORS:
    g = t[t.sector == s].sort_values(YR); yr = g[YR].values
    for c, _ in COLS:
        v = g[c].values
        for y0, y1 in ((yr.min(), BREAK_YEAR - 1), (BREAK_YEAR, yr.max())):
            rows.append(dict(sector=s, component=c, era=f"{int(y0)}-{int(y1)}",
                             mean_abs=np.nanmean(np.abs(v[(yr >= y0) & (yr <= y1)]))))
e = pd.DataFrame(rows)
e.to_csv(os.path.join(TABLES_DIR, "t32_component_size_era.csv"), index=False)
print(e.pivot_table(index=["sector", "component"], columns="era", values="mean_abs").round(3).to_string())

nr, nc = len(PLOT_SECTORS), len(PLOT_COLS)
fig, axes = plt.subplots(nr, nc, figsize=(9.0, 1.6 * nr + 0.6), sharex=True, squeeze=False)
for i, s in enumerate(PLOT_SECTORS):
    g = t[t.sector == s].sort_values(YR); yr = g[YR].values
    lim = 1.08 * np.nanmax(np.abs(g[[c for c, _ in PLOT_COLS]].values))
    for j, (c, lab) in enumerate(PLOT_COLS):
        ax = axes[i, j]; v = g[c].values
        ax.bar(yr, v, width=0.8, color=np.where(v >= 0, S.POS, S.NEG), lw=0)
        ax.axhline(0, color=S.INK, lw=0.6)
        ax.axvline(BREAK_YEAR - 0.5, color=S.GREY, lw=0.8, ls=(0, (2, 2)))
        ax.set_ylim(-lim, lim)
        for y0, y1 in ((yr.min(), BREAK_YEAR - 1), (BREAK_YEAR, yr.max())):
            m = np.nanmean(np.abs(v[(yr >= y0) & (yr <= y1)]))
            ax.plot([y0, y1], [0.86 * lim] * 2, color=S.INK, lw=1.4, solid_capstyle="butt")
            ax.text((y0 + y1) / 2, 0.88 * lim, f"{m:.2f}", ha="center", va="bottom", fontsize=7.5, color=S.INK)
        if j > 0:
            ax.tick_params(labelleft=False)
        if i == 0:
            ax.set_title(lab, pad=12)
        if j == 0:
            ax.set_ylabel(ROWLAB[s], labelpad=6)
        if i == nr - 1:
            ax.set_xticks([1980, 2000, 2020])
fig.supylabel("Cycle-mean contribution (10$^6$ km$^2$)", fontsize=9, color=S.INK, x=0.005)
fig.tight_layout(rect=[0.02, 0, 1, 1], h_pad=0.9, w_pad=0.4)
S.save_figure(fig, "ch3_fig04_components_by_cycle")
