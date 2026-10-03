#!/usr/bin/env python3
"""
fig_03_components_by_cycle.py -- Fig. 3 (replaces the old Figs 3 and 4): each
cycle's departure from the invariant cycle and three of its components,
drawn SEPARATELY rather than stacked.

Rows = sectors (Weddell, A-B, Ross, EA, KH). Circumpolar total (sum of these
five) is omitted from the plotted grid -- it adds no information beyond the
five rows -- but its numbers are still in the companion CSV below.
Columns = the departure from the invariant cycle (what an extent anomaly
sees), then trend, amplitude, and phase. Raw anomaly (the 4th component, a
residual) is also omitted from the plotted grid: checked against
t32_component_size_era.csv, it is the smallest and flattest term in every
sector and era (0.01-0.06 x10^6 km^2 throughout) -- unlike phase, kept
deliberately: phase beats raw anomaly in every sector/era and is GROWING
post-2016 in several (Ross 0.069->0.099, Weddell 0.055->0.087, circumpolar
0.121->0.133), so it's part of the post-2016 story this figure is making,
not noise to cut. Dropping circumpolar + raw anomaly together takes the
grid from 30 panels (6x5) to 20 (5x4).
One bar per cycle (21 Feb to 20 Feb), positive red / negative blue. All
panels in a row share one y-axis, so the components can be compared with
each other and with the total directly. In each panel, two short horizontal
lines give the mean SIZE of that component (mean |bar|) over 1979-2015 and
2016-2024, with the values printed, which replaces the era-share figure
(old Fig. 4) and the NET/GROSS distinction: the figure shows the sizes
themselves.

Phase is small in these bars for a reason the caption should state: a shift
in timing adds ice in the advance and removes it in the retreat, so it
largely cancels in the mean over a cycle. Fig. 10 shows it day by day.

DEPENDENCIES (added 2026-10-01, part of the per-figure cblind/dual-export pass)
--------------------------------------------------------------------------
Data:   none beyond the table below -- no direct NetCDF/pipeline read.
Tables: reads results/ch3/tables/t32_attribution_by_cycle.csv (written by
        fig_03-04_attribution_annual.py -- this script exits with a clear
        message if that file is missing).
        Writes results/ch3/tables/t32_component_size_era.csv -- era-mean
        |contribution| for ALL 6 sectors x ALL 5 components, independent of
        the 5x4 grid actually plotted (see above).
Colors: positive/negative bars now draw from ch3_config.CBLIND_COLORS
        (vermillion / blue) instead of local hardcoded hex.

Reads  results/ch3/tables/t32_attribution_by_cycle.csv (from fig_03-04_attribution_annual.py)
Writes results/ch3/figures/fig03_components_by_cycle.png              (white bg, manuscript)
       results/ch3/figures/fig03_components_by_cycle_presentation.png (transparent bg, slides)
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
from ch3_plot import save
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
# Plotted subset (2026-10-01): circumpolar + raw anomaly dropped from the
# grid -- see the DEPENDENCIES note in the module docstring for why. Both
# are still computed and written to t32_component_size_era.csv in full.
PLOT_SECTORS = [s for s in SECTORS if s != "SIE_circumpolar"]
PLOT_COLS = [(c, lab) for c, lab in COLS if c != "residual_apac"]
LABEL = {"SIE_Weddell": "Weddell", "SIE_Amundsen_Bellingshausen": "Amundsen-\nBellingshausen",
         "SIE_Ross": "Ross", "SIE_East_Antarctica": "East\nAntarctica",
         "SIE_King_Haakon": "King Haakon", "SIE_circumpolar": "Circumpolar\ntotal"}
# Colorblind-safe (Okabe-Ito), replacing local hardcoded hex (2026-10-01).
# Fixed 2026-10-03: CBLIND_COLORS isn't exported from ch3_config (it's a
# plain list defined in ch3_style.py, not a dict there either) -- this would
# have raised ImportError on the next run. ch3_style.VERMILLION/BLUE are the
# actual named, colorblind-safe constants other scripts already use.
POS, NEG, INK = ch3_style.VERMILLION, ch3_style.BLUE, "0.35"

# Landscape width (2026-10-01, confirmed OK with the Graduate Division): this
# figure is placed as a landscape page in the thesis -- the 20-panel grid
# (47 years x 4 components x 5 sectors) was too cramped at the portrait
# ch3_config.TEXT_WIDTH_IN (6.5in). Sized the same way TEXT_WIDTH_IN is
# (page width minus 1in margins each side), on an 11in landscape page.
LANDSCAPE_WIDTH_IN = 9.0

# -- stats pass: ALL 6 sectors x ALL 5 components, independent of what's
# plotted below (2026-10-01) -- t32_component_size_era.csv is unaffected by
# which panels end up in the figure.
rows = []
for s in SECTORS:
    g = t[t.sector == s].sort_values(YR)
    yr = g[YR].values
    for c, lab in COLS:
        v = g[c].values
        for y0, y1 in ((yr.min(), BREAK_YEAR - 1), (BREAK_YEAR, yr.max())):
            m = np.nanmean(np.abs(v[(yr >= y0) & (yr <= y1)]))
            rows.append(dict(sector=s, component=c, era=f"{int(y0)}-{int(y1)}", mean_abs=m))
e = pd.DataFrame(rows)
e.to_csv(os.path.join(TABLES_DIR, "t32_component_size_era.csv"), index=False)
print(e.pivot_table(index=["sector", "component"], columns="era", values="mean_abs").round(3).to_string())

# -- plot pass: PLOT_SECTORS x PLOT_COLS only (circumpolar + raw anomaly
# dropped from the grid, 2026-10-01 -- full stats above are unaffected).
nr, nc = len(PLOT_SECTORS), len(PLOT_COLS)
# Landscape figure (2026-10-01, confirmed OK with the Graduate Division) --
# LANDSCAPE_WIDTH_IN above, not the portrait TEXT_WIDTH_IN. save() will
# still print its "wider than TEXT_WIDTH_IN" warning below since that check
# only knows about the portrait width -- expected and fine to ignore here.
# Place this figure's page as landscape when inserting it into the thesis.
fig, axes = plt.subplots(nr, nc, figsize=(LANDSCAPE_WIDTH_IN, 1.75 * nr + 0.8),
                         sharex=True, squeeze=False)
bold = ch3_style.bold_font_properties(size=10.5)
for i, s in enumerate(PLOT_SECTORS):
    g = t[t.sector == s].sort_values(YR)
    yr = g[YR].values
    lim = 1.08 * np.nanmax(np.abs(g[[c for c, _ in PLOT_COLS]].values))
    for j, (c, lab) in enumerate(PLOT_COLS):
        ax = axes[i, j]
        v = g[c].values
        ax.bar(yr, v, width=0.8, color=np.where(v >= 0, POS, NEG), lw=0)
        ax.axhline(0, color="0.2", lw=0.6)
        ax.axvline(BREAK_YEAR - 0.5, color="0.5", lw=0.8, ls=(0, (2, 2)))
        ax.set_ylim(-lim, lim)
        # era mean size, drawn at the top of the panel as a bracket-like tick
        # (text bumped 7 -> 9.5pt, 2026-10-01, now that the grid is smaller)
        for y0, y1 in ((yr.min(), BREAK_YEAR - 1), (BREAK_YEAR, yr.max())):
            m = np.nanmean(np.abs(v[(yr >= y0) & (yr <= y1)]))
            ax.plot([y0, y1], [0.86 * lim, 0.86 * lim], color="0.15", lw=1.4, solid_capstyle="butt")
            ax.text((y0 + y1) / 2, 0.86 * lim * 1.02, f"{m:.2f}", ha="center", va="bottom",
                    fontsize=9.5, color="0.15")
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
out = save(fig, "fig03_components_by_cycle.png", dpi=250, sync=False, presentation=True)
print("wrote", out)