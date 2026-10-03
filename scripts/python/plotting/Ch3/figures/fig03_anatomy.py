"""
fig03_anatomy.py  —  Handcock & Raphael (2020) Fig 7a idiom, per sector, per year.

For a given year, one panel per sector, over day of year:
  - invariant annual cycle (light blue, centred; the cycle being decomposed)
  - trend component (green), amplitude component (navy), phase component (red)
  - raw anomaly (thin black)  = Extent - invariant cycle = anomaly_from_iac
  - estimated anomaly (orange) = trend + amplitude + phase = est_anomaly
The gap between raw (black) and estimated (orange) is the residual.

Produces one figure per year in YEARS. Set YEARS = range(YEAR_MIN, YEAR_MAX+1)
to generate all years for browsing; the paper uses 2016 and 2023.

Run from scripts/python/plotting/Ch3/figures/ :  python fig03_anatomy.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__)); import sys; sys.path.insert(0, HERE)
from ch3_config import OUTPUT_DIR, YEAR_MIN, YEAR_MAX
from ch3_config import DAILY_CSV  # canonical daily_fitted.csv (old _E fallback removed 2026-10-04)

YEARS = [2016, 2023]                       # set to range(YEAR_MIN, YEAR_MAX+1) to browse all
# The invariant cycle swings ~8x the components; if it sets the y-axis it flattens
# everything. So each panel is scaled to the components (native 10^6 km^2) and the
# invariant cycle is drawn only as light, clipped background context. Set False to drop it.
SHOW_INVARIANT = True
SECTORS = ["SIE_circumpolar", "SIE_Amundsen_Bellingshausen", "SIE_East_Antarctica",
           "SIE_Weddell", "SIE_Ross", "SIE_King_Haakon"]
LABELS = {"SIE_Amundsen_Bellingshausen": "Amundsen–Bellingshausen", "SIE_East_Antarctica": "East Antarctica",
          "SIE_Weddell": "Weddell", "SIE_Ross": "Ross", "SIE_King_Haakon": "King Haakon", "SIE_circumpolar": "Circumpolar"}
C_IAC, C_TR, C_AMP, C_PH, C_RAW, C_EST = "#8fc7e8", "#2ca02c", "#12128c", "#e01111", "k", "#f5a623"

d = pd.read_csv(DAILY_CSV)
mon = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]; ml = list("JFMAMJJASOND")
# centre the invariant cycle per sector (seasonal shape as an anomaly)
iac_mean = d.groupby("sector")["iac_notrend"].transform("mean")
d = d.assign(iac_centred=d["iac_notrend"] - iac_mean)

def make_year(yr):
    dy = d[d.Year == yr]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    axes = axes.ravel()
    for k, sec in enumerate(SECTORS):
        ax = axes[k]; a = dy[dy.sector == sec].sort_values("DOY")
        x = a.DOY.values
        # the four components, raw anomaly, and estimated anomaly: native 10^6 km^2
        comp = [a.trend_component.values, a.amplitude_component.values,
                a.phase_component.values, a.anomaly_from_iac.values, a.est_anomaly.values]
        ax.plot(x, a.trend_component.values, color=C_TR, lw=2.2, zorder=3)
        ax.plot(x, a.amplitude_component.values, color=C_AMP, lw=2.2, zorder=3)
        ax.plot(x, a.phase_component.values, color=C_PH, lw=2.4, zorder=3)
        ax.plot(x, a.anomaly_from_iac.values, color=C_RAW, lw=0.8, zorder=4)
        ax.plot(x, a.est_anomaly.values, color=C_EST, lw=1.8, zorder=5)
        # scale the panel to the COMPONENTS, not the invariant cycle
        lim = 1.15 * np.nanmax(np.abs(np.concatenate(comp)))
        ax.set_ylim(-lim, lim)
        if SHOW_INVARIANT:   # light background context, clipped to the component scale
            ax.plot(x, a.iac_centred.values, color=C_IAC, lw=2.4, alpha=0.55,
                    solid_capstyle="round", zorder=1, clip_on=True)
        ax.axhline(0, color="k", lw=0.6, ls=(0, (4, 3)))
        ax.set_title(LABELS.get(sec, sec), fontsize=11, pad=3)
        ax.set_xlim(1, 365); ax.set_xticks(mon); ax.set_xticklabels(ml, fontsize=8)
        ax.tick_params(labelsize=8)
        if k % 3 == 0: ax.set_ylabel("anomaly for sea ice extent\n(10$^6$ km$^2$)", fontsize=9)
    leg = [Line2D([], [], color=C_IAC, lw=3.2, label="invariant annual cycle"),
           Line2D([], [], color=C_TR, lw=2.2, label="trend"), Line2D([], [], color=C_AMP, lw=2.2, label="amplitude"),
           Line2D([], [], color=C_PH, lw=2.4, label="phase"), Line2D([], [], color=C_RAW, lw=0.8, label="raw anomaly"),
           Line2D([], [], color=C_EST, lw=1.8, label="estimated anomaly")]
    fig.legend(handles=leg, ncol=6, loc="lower center", frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"Anatomy of the {yr} annual cycle by sector", fontsize=12, y=0.99)
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    out = os.path.join(OUTPUT_DIR, f"fig03_anatomy_{yr}.png")
    fig.savefig(out, dpi=170, bbox_inches="tight"); plt.close(fig); print("wrote", out)

for y in YEARS:
    make_year(y)
