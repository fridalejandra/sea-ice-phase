#!/usr/bin/env python3
"""
fig_s04_record_noise.py -- Fig. S4 (rebuilt): why daily statistics stop in 2023.

Three panels, one quantity each, all six series on each panel (five sectors thin,
circumpolar total thick), 1988-2025:
  (a) year-by-year SD of the day-to-day change in extent, relative to its 1988-2015 mean
  (b) year-by-year SD of the raw anomaly, relative to its 1988-2015 mean
  (c) year-by-year lag-1 autocorrelation of the raw anomaly (day-to-day correlation)
(a) and (b) on a log scale so a doubling and a halving look the same size, with
the 2024-2025 years shaded. The point the caption makes: in 2024-25 (a) jumps in
every sector at once, (b) barely moves, and (c) drops -- which is what added
day-to-day noise in the record does (noise lowers the day-to-day correlation
without making the anomaly larger), not what a change in the ice would do.

Reads  DAILY_CSV (sector, Date, Year, Extent, residual_apac; period == FULL)
Writes results/ch3/figures/fig_s04_record_noise.png
       results/ch3/tables/t_s04_record_noise.csv
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ch3_config import DAILY_CSV, OUTPUT_DIR, TABLES_DIR, SECTORS
import ch3_style

Y0, Y1, REF1 = 1988, 2025, 2015
TITLE = {"SIE_Weddell": "Weddell", "SIE_Amundsen_Bellingshausen": "Amundsen-Bellingshausen",
         "SIE_Ross": "Ross", "SIE_East_Antarctica": "East Antarctica",
         "SIE_King_Haakon": "King Haakon", "SIE_circumpolar": "Circumpolar total"}
# sector identity is not the point here: sectors light grey, the total black
COL = {s: ("0.1" if "circ" in s else "0.72") for s in SECTORS}
INK = "0.35"

d = pd.read_csv(DAILY_CSV, parse_dates=["Date"])
if "period" in d.columns:
    d = d[d["period"] == "FULL"]
ext = next(c for c in ("Extent", "extent", "SIE") if c in d.columns)
d = d[d["Year"].between(Y0, Y1)].sort_values(["sector", "Date"])

rows = []
for sec in SECTORS:
    g = d[d.sector == sec].set_index("Date")
    for y, gy in g.groupby("Year"):
        e = gy[ext]
        step = e.diff()[gy.index.to_series().diff().dt.days == 1]      # consecutive days only
        r = gy["residual_apac"]
        r0, r1 = r.values[:-1], r.values[1:]
        ok = (np.diff(gy.index.values).astype("timedelta64[D]").astype(int) == 1)
        rows.append(dict(sector=sec, Year=y, sd_step=step.std(), sd_raw=r.std(),
                         lag1=np.corrcoef(r0[ok], r1[ok])[0, 1] if ok.sum() > 30 else np.nan))
t = pd.DataFrame(rows)
for c in ("sd_step", "sd_raw"):
    ref = t[t.Year <= REF1].groupby("sector")[c].mean()
    t[c + "_rel"] = t[c] / t["sector"].map(ref)
t.to_csv(os.path.join(TABLES_DIR, "t_s04_record_noise.csv"), index=False)

panels = [("sd_step_rel", "(a)  Day-to-day change in extent", "SD relative to 1988–2015", True),
          ("sd_raw_rel", "(b)  Raw anomaly", "SD relative to 1988–2015", True),
          ("lag1", "(c)  Raw anomaly, day-to-day correlation", "correlation", False)]
fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9))
bold = ch3_style.bold_font_properties(size=10)
for ax, (c, title, ylab, logy) in zip(axes, panels):
    ax.axvspan(2023.5, Y1 + 0.5, color="0.92", zorder=0)
    ax.axvline(2007.5, color="0.65", lw=0.8, ls=(0, (1, 2)), zorder=1)   # SSM/I -> SSMIS
    if logy:
        ax.axhline(1, color="0.6", lw=0.8, zorder=1)
    for sec in SECTORS:
        s = t[t.sector == sec]
        circ = "circ" in sec
        ax.plot(s.Year, s[c], color=COL[sec], lw=2.2 if circ else 1.0,
                zorder=3 if circ else 2, label=("Circumpolar total" if circ else
                                               ("Each sector" if sec == SECTORS[0] else None)))
    if logy:
        ax.set_yscale("log", base=2)
        ticks = [0.5, 1, 2, 4, 8]
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_yticklabels([f"{v:g}×" for v in ticks])
        ax.set_ylim(0.4, max(8.5, 1.1 * np.nanmax(t[c])))
    ax.set_xlim(Y0 - 0.5, Y1 + 0.5)
    ax.set_title(title, loc="left", fontproperties=bold, color="0.1")
    ax.set_ylabel(ylab, fontsize=8.5, color=INK)
    ax.tick_params(labelsize=8, colors=INK, length=2.5)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].text(2024, axes[0].get_ylim()[1] * 0.92, "2024–25", ha="center", va="top", fontsize=8, color=INK)
axes[0].text(2007.2, axes[0].get_ylim()[1] * 0.92, "sensor change\n(2008)", ha="right", va="top",
             fontsize=7.5, color=INK)
axes[2].legend(frameon=False, fontsize=7.5, loc="lower left")
fig.tight_layout(w_pad=2)
out = os.path.join(OUTPUT_DIR, "fig_s04_record_noise.png")
fig.savefig(out, dpi=250, bbox_inches="tight")
plt.close(fig)

print(f"wrote {out}")
late = t[t.Year >= 2024].copy()
late["sector"] = late.sector.map(TITLE)
print("\n2024-2025 (relative to 1988-2015 means; lag1 absolute):")
print(late[["sector", "Year", "sd_step_rel", "sd_raw_rel", "lag1"]].round(2).to_string(index=False))
base = t[t.Year <= REF1].groupby("sector")["lag1"].mean().rename(index=TITLE)
# trend in the day-to-day change before the record problem: replaces the GAMLSS volatility trend
from scipy.stats import linregress
print("\nTrend in the SD of the day-to-day change, 1988-2023 (% of the 1988-2015 mean per decade):")
for sec in SECTORS:
    s_ = t[(t.sector == sec) & (t.Year <= 2023)]
    f = linregress(s_.Year, 100 * s_.sd_step_rel)
    print(f"   {TITLE[sec]:24s} {10 * f.slope:+5.1f} % per decade   p = {f.pvalue:.3f}")
print("\n1988-2015 mean lag-1 by sector:", ", ".join(f"{k} {v:.3f}" for k, v in base.items()))