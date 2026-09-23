#!/usr/bin/env python3
"""
fig_s03_residual_sd_by_doy.py -- Fig. S3 without the GAMLSS: how much the raw
anomaly varies on each day of the year, 1988-2015 against 2016-2023.

For each sector and period: the standard deviation across years of the raw
anomaly on each day of year, pooled over a +/-15-day window so the curve is
smooth without a model. Shading is a 95 % interval from resampling whole years.

Difference from the GAMLSS version: no term for the satellite-sensor change
(SSM/I -> SSMIS in 2008). Both periods straddle or follow it (1988-2015 spans
it, 2016-2023 is all SSMIS), so a sensor effect would appear as a difference
between the curves; say so in the caption if the curves differ.

Reads  daily_fitted.csv (period == FULL) via ch3_data
Writes results/ch3/figures/figS03_residual_sd_by_doy.png
       results/ch3/tables/tS03_residual_sd_by_doy.csv
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ch3_data as D
from ch3_config import SECTORS, TABLES_DIR, OUTPUT_DIR
import ch3_style

HALF = 15
N_BOOT = int(os.environ.get("NBOOT", "200"))   # ~1-2 min; raise for the final figure
PERIODS = [("1988–2015", 1988, 2015, "#2a78d6"), ("2016–2023", 2016, 2023, "#eb6834")]
TITLE = {"SIE_Weddell": "Weddell", "SIE_Amundsen_Bellingshausen": "Amundsen-Bellingshausen",
         "SIE_Ross": "Ross", "SIE_East_Antarctica": "East Antarctica",
         "SIE_King_Haakon": "King Haakon", "SIE_circumpolar": "Circumpolar total"}
MONTH_DOY = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
MONTH_LAB = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]

d = D.load_daily(period="FULL")
d["Date"] = pd.to_datetime(d["Date"])
d = d[(d.Year >= 1988) & (d.Year <= 2023) & (d.DOY <= 365)]
# remove each sector's mean seasonal cycle of the residual first
d["ra"] = d["residual_apac"] - d.groupby(["sector", "DOY"])["residual_apac"].transform("mean")


def sd_curve(M):
    """M: years x 365 array. SD across years, pooled over a circular +/-HALF window."""
    out = np.full(365, np.nan)
    for k in range(365):
        idx = [(k + j) % 365 for j in range(-HALF, HALF + 1)]
        x = M[:, idx]
        x = x - np.nanmean(x, axis=0)          # anomaly about each day's mean across years
        out[k] = np.sqrt(np.nanmean(x ** 2))
    return out


rows = []
fig, axes = plt.subplots(2, 3, figsize=(12, 6.2), sharex=True)
bold = ch3_style.bold_font_properties(size=11)
rng = np.random.default_rng(3)
for k, s in enumerate(SECTORS):
    ax = axes.ravel()[k]
    g = d[d.sector == s]
    for lab, y0, y1, col in PERIODS:
        w = g[(g.Year >= y0) & (g.Year <= y1)].pivot_table(index="Year", columns="DOY", values="ra")
        w = w.reindex(columns=range(1, 366))
        M = w.values
        c = sd_curve(M)
        boots = np.array([sd_curve(M[rng.integers(0, len(M), len(M))]) for _ in range(N_BOOT)])
        lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
        x = np.arange(1, 366)
        ax.fill_between(x, lo, hi, color=col, alpha=0.18, lw=0)
        ax.plot(x, c, color=col, lw=2, label=lab)
        rows += [dict(sector=s, period=lab, doy=int(i), sd=float(v), lo=float(a), hi=float(b))
                 for i, v, a, b in zip(x, c, lo, hi)]
    ax.set_title(f"({chr(97 + k)})  {TITLE.get(s, s)}", loc="left", fontproperties=bold, color="0.1")
    ax.set_xticks(MONTH_DOY); ax.set_xticklabels(MONTH_LAB)
    ax.set_xlim(1, 365); ax.set_ylim(bottom=0)
    ax.spines[["top", "right"]].set_visible(False)
    if k % 3 == 0:
        ax.set_ylabel("SD of raw anomaly (10$^6$ km$^2$)", fontsize=9)
axes.ravel()[0].legend(frameon=False, fontsize=9, loc="upper left")
fig.tight_layout()
out = os.path.join(OUTPUT_DIR, "figS03_residual_sd_by_doy.png")
fig.savefig(out, dpi=220)
pd.DataFrame(rows).to_csv(os.path.join(TABLES_DIR, "tS03_residual_sd_by_doy.csv"), index=False)
print("wrote", out)
