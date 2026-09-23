#!/usr/bin/env python3
"""
fig_03_components_by_cycle.py -- Fig. 3 (replaces the old Figs 3 and 4): each
cycle's departure from the invariant cycle and its four components, drawn
SEPARATELY rather than stacked.

Rows = sectors (Weddell, A-B, Ross, EA, KH, circumpolar).
Columns = the departure from the invariant cycle (what an extent anomaly
sees), then trend, amplitude, phase and raw anomaly. One bar per cycle
(21 Feb to 20 Feb), positive red / negative blue. All panels in a row share
one y-axis, so the components can be compared with each other and with the
total directly. In each panel, two short horizontal lines give the mean SIZE
of that component (mean |bar|) over 1979-2015 and 2016-2024, with the values
printed, which replaces the era-share figure (old Fig. 4) and the NET/GROSS
distinction: the figure shows the sizes themselves.

Phase is small in these bars for a reason the caption should state: a shift
in timing adds ice in the advance and removes it in the retreat, so it
largely cancels in the mean over a cycle. Fig. 10 shows it day by day.

Reads  results/ch3/tables/t32_attribution_by_cycle.csv (from fig_03-04_attribution_annual.py)
Writes results/ch3/figures/fig03_components_by_cycle.png
       results/ch3/tables/t32_component_size_era.csv
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ch3_config import SECTORS, TABLES_DIR, OUTPUT_DIR, BREAK_YEAR
import ch3_style

SRC = os.path.join(TABLES_DIR, "t32_attribution_by_cycle.csv")
if not os.path.exists(SRC):
    sys.exit(f"{SRC} not found; run fig_03-04_attribution_annual.py first")
t = pd.read_csv(SRC)
YR = "Year" if "Year" in t.columns else "cycle"

COLS = [("anomaly_from_iac", "Extent anomaly"),
        ("trend_component", "Trend"),
        ("amplitude_component", "Amplitude"),
        ("phase_component", "Phase"),
        ("residual_apac", "Raw anomaly")]
LABEL = {"SIE_Weddell": "Weddell", "SIE_Amundsen_Bellingshausen": "Amundsen-\nBellingshausen",
         "SIE_Ross": "Ross", "SIE_East_Antarctica": "East\nAntarctica",
         "SIE_King_Haakon": "King Haakon", "SIE_circumpolar": "Circumpolar\ntotal"}
POS, NEG, INK = "#d6604d", "#4393c3", "0.35"

nr, nc = len(SECTORS), len(COLS)
fig, axes = plt.subplots(nr, nc, figsize=(2.35 * nc + 1.0, 1.75 * nr + 0.8),
                         sharex=True, squeeze=False)
bold = ch3_style.bold_font_properties(size=10.5)
rows = []
for i, s in enumerate(SECTORS):
    g = t[t.sector == s].sort_values(YR)
    yr = g[YR].values
    lim = 1.08 * np.nanmax(np.abs(g[[c for c, _ in COLS]].values))
    for j, (c, lab) in enumerate(COLS):
        ax = axes[i, j]
        v = g[c].values
        ax.bar(yr, v, width=0.8, color=np.where(v >= 0, POS, NEG), lw=0)
        ax.axhline(0, color="0.2", lw=0.6)
        ax.axvline(BREAK_YEAR - 0.5, color="0.5", lw=0.8, ls=(0, (2, 2)))
        ax.set_ylim(-lim, lim)
        # era mean size, drawn at the top of the panel as a bracket-like tick
        for (y0, y1), yy in (((yr.min(), BREAK_YEAR - 1), 0.86), ((BREAK_YEAR, yr.max()), 0.86)):
            m = np.nanmean(np.abs(v[(yr >= y0) & (yr <= y1)]))
            ax.plot([y0, y1], [yy * lim, yy * lim], color="0.15", lw=1.4, solid_capstyle="butt")
            ax.text((y0 + y1) / 2, yy * lim * 1.02, f"{m:.2f}", ha="center", va="bottom",
                    fontsize=7, color="0.15")
            rows.append(dict(sector=s, component=c, era=f"{int(y0)}-{int(y1)}", mean_abs=m))
        ax.tick_params(labelsize=7.5, colors=INK, length=2.5)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(INK)
        ax.spines[["top", "right"]].set_visible(False)
        if j > 0:
            ax.tick_params(labelleft=False)
        if i == 0:
            ax.set_title(lab, fontproperties=bold, color="0.1", pad=14)
        if j == 0:
            ax.set_ylabel(LABEL.get(s, s), fontproperties=bold, color="0.1", labelpad=6)
        if i == nr - 1:
            ax.set_xticks([1980, 2000, 2020])
fig.text(0.005, 0.5, "cycle-mean contribution (10$^6$ km$^2$)", rotation=90, va="center",
         fontsize=9, color=INK)
fig.tight_layout(rect=[0.02, 0, 1, 1], h_pad=0.9, w_pad=0.4)
out = os.path.join(OUTPUT_DIR, "fig03_components_by_cycle.png")
fig.savefig(out, dpi=250, bbox_inches="tight")
plt.close(fig)
e = pd.DataFrame(rows)
e.to_csv(os.path.join(TABLES_DIR, "t32_component_size_era.csv"), index=False)
print(f"wrote {out}")
print(e.pivot_table(index=["sector", "component"], columns="era", values="mean_abs").round(3).to_string())