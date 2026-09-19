"""
fig11_trend_volatility.py — the low-frequency trend term, per sector.
======================================================================
What shape is the GAM's low-frequency term s(t)? A step at 2016, a decline, or a
reversal? Plotted as the annual mean of trend_component per sector, with the
2016 break.

Daily statistics start at YEAR_START (1980) because 1979's residual carries a
spline edge effect (sd ~1.5x later years).

VOLATILITY (formerly panels (b)/(c) here, and t34b_volatility_annual.csv /
t34b_volatility_2016_ratio.csv) has been removed from this script: it's
superseded by the seasonal-gamlss analysis in 05_volatility_gamlss.R
(t34c_volatility_gamlss_post2016.csv, t34c_volatility_seasonal_curves.csv,
fig12_volatility_seasonal.py), which controls for season and sensor and
quantifies uncertainty with a year-block bootstrap instead of a Welch t-test
on annual values. One of the two old measures (mean_garch_sd) also no longer
has a live input: 01_fit_apac.R's daily `volatility` column (the old per-day
GARCH conditional SD) is retired/NA as of the dual-period consolidation.

Outputs
    results/ch3/figures/fig11_trend_volatility.png
    results/ch3/tables/t34b_trend_annual.csv               sector, Year, trend (annual mean of s(t))
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import ch3_data as D
from ch3_config import (SECTORS, SECTOR_LABELS, SECTOR_COLORS, TABLES_DIR,
                        BREAK_YEAR, YEAR_START, YEAR_MAX)
from ch3_plot import mark_break, zero_line, year_axis, panel_letters, save, figsize

print("fig11 — trend term")
daily = D.load_daily()
D.summary(daily=daily)
d = daily[daily["Year"].between(YEAR_START, YEAR_MAX)].copy()

tr = (d.groupby(["sector", "Year"])["trend_component"].mean().reset_index()
        .rename(columns={"trend_component": "trend"}))
tr.to_csv(os.path.join(TABLES_DIR, "t34b_trend_annual.csv"), index=False)
print(tr.round(3).to_string(index=False))

# ── figure: 1 row (trend) x 2 (sectors overlaid | circumpolar on its own axis)
fig, axes = plt.subplots(1, 2, figsize=figsize("row2"), sharex=True, gridspec_kw=dict(width_ratios=[2.2, 1]))
for j, secs in enumerate(([s for s in SECTORS if s != "SIE_circumpolar"], ["SIE_circumpolar"])):
    ax = axes[j]
    for sec in secs:
        g = tr[tr.sector == sec].sort_values("Year")
        ax.plot(g.Year, g.trend, color=SECTOR_COLORS[sec], lw=1.8, label=SECTOR_LABELS[sec])
    zero_line(ax)
    mark_break(ax); year_axis(ax, YEAR_START, YEAR_MAX)
    ax.spines[["top", "right"]].set_visible(False)
    if j == 0: ax.set_ylabel("s(t), annual mean (10⁶ km²)", fontsize=9)
    ax.set_title("sectors" if j == 0 else "circumpolar", fontsize=10, fontweight="bold")
axes[0].legend(fontsize=8, frameon=False, ncol=3, loc="upper left")
panel_letters(axes.ravel())
fig.suptitle("Low-frequency trend term, by sector", fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.90])
save(fig, "fig11_trend_volatility.png", sync=False)