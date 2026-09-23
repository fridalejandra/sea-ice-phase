#!/usr/bin/env python3
"""
fig_s02_annual_min_max_trend.py -- Fig. S2 (restyled): annual summer minimum
and winter maximum extent by sector, 1979-2025, with a linear trend.

Layout: six rows (sectors) x two columns (minimum, maximum). Points coloured by
period as in Figs. 8 and S3 (1979-2015 blue, 2016-2025 orange), dotted line at
2016, dashed grey trend line, trend per decade in the corner (* p < 0.05).

LABEL CHECK. Two renders of this figure had the Weddell and Amundsen-
Bellingshausen rows the other way round (the v4 caption says the Weddell
minimum declines significantly; the v4 render puts that panel under "ABS").
Before plotting, each raw daily column is correlated with the observed extent
of every sector in daily_fitted.csv. Each column should match its own sector
best (r ~ 1); if not, the script stops and prints the matrix.

Reads  DAILY_RAW_CSV (time, SIE_<sector> columns), DAILY_CSV (sector, Date, Extent)
Writes results/ch3/figures/fig_s02_annual_min_max_trend.png
       results/ch3/tables/t_s02_annual_min_max_trend.csv
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ch3_config import DAILY_RAW_CSV, DAILY_CSV, OUTPUT_DIR, TABLES_DIR, SECTORS, BREAK_YEAR
import ch3_style

Y0, Y1 = 1979, 2025
TITLE = {"SIE_Weddell": "Weddell", "SIE_Amundsen_Bellingshausen": "Amundsen-Bellingshausen",
         "SIE_Ross": "Ross", "SIE_East_Antarctica": "East Antarctica",
         "SIE_King_Haakon": "King Haakon", "SIE_circumpolar": "Circumpolar total"}
PRE, POST, INK = "#2a78d6", "#eb6834", "0.35"

raw = pd.read_csv(DAILY_RAW_CSV, parse_dates=["time"])
raw["Year"] = raw["time"].dt.year
raw = raw[raw.Year.between(Y0, Y1)]

# ---- label check against the APAC input ---------------------------------
fit = pd.read_csv(DAILY_CSV, parse_dates=["Date"])
if "period" in fit.columns:
    fit = fit[fit["period"] == "FULL"]
ext = next((c for c in ("Extent", "extent", "SIE") if c in fit.columns), None)
if ext:
    W = fit.pivot_table(index="Date", columns="sector", values=ext)
    W.columns = [f"fit::{c}" for c in W.columns]
    R = raw.set_index(raw["time"].dt.normalize())[[s for s in SECTORS if s in raw.columns]]
    J = R.join(W, how="inner").dropna()
    M = pd.DataFrame({s: {c[5:]: np.corrcoef(J[s], J[c])[0, 1] for c in W.columns}
                      for s in R.columns}).T
    best = M.idxmax(axis=1)
    bad = [s for s in R.columns if best[s] != s]
    print("label check: raw column -> best-matching sector in daily_fitted.csv")
    for s in R.columns:
        print(f"   {s:30s} -> {best[s]:30s} r = {M.loc[s, best[s]]:.3f}")
    if bad:
        print("\n" + M.round(3).to_string())
        sys.exit(f"\nSTOP: {bad} match another sector better than their own name. "
                 "The two files disagree about which column is which; fix before plotting.")
else:
    print("no Extent column in daily_fitted.csv; label check skipped")

# ---- figure ----------------------------------------------------------------
bold = ch3_style.bold_font_properties(size=9.5)
rows = []
fig, axes = plt.subplots(len(SECTORS), 2, figsize=(8.2, 1.75 * len(SECTORS) + 0.6), sharex=True)
for i, sec in enumerate(SECTORS):
    g = raw.groupby("Year")[sec].agg(["min", "max"])
    for j, (col, head) in enumerate([("min", "Summer minimum"), ("max", "Winter maximum")]):
        ax = axes[i, j]
        yr, v = g.index.values, g[col].values
        fitl = linregress(yr, v)
        rows.append(dict(sector=TITLE[sec], extremum=col, n=len(yr), mean=v.mean(),
                         slope_per_decade=fitl.slope * 10, p=fitl.pvalue))
        post = yr >= BREAK_YEAR
        ax.axvline(BREAK_YEAR - 0.5, color="0.6", lw=0.8, ls=(0, (2, 2)), zorder=1)
        ax.plot(yr, fitl.intercept + fitl.slope * yr, color="0.45", lw=1.1, ls="--", zorder=2)
        ax.scatter(yr[~post], v[~post], s=13, color=PRE, lw=0, zorder=3)
        ax.scatter(yr[post], v[post], s=13, color=POST, lw=0, zorder=3)
        star = "*" if fitl.pvalue < 0.05 else ""
        lab = f"{fitl.slope * 10:+.2f}".replace("+0.00", "0.00").replace("-0.00", "0.00").replace("-", "−")
        ax.set_title(f"{lab} per decade{star}", loc="right", fontsize=8, color="0.25", pad=3)
        pad = 0.12 * (v.max() - v.min())
        ax.set_ylim(v.min() - pad, v.max() + pad)
        ax.tick_params(labelsize=7.5, colors=INK, length=2.5)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(INK)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(f"({chr(97 + 2 * i + j)})  {TITLE[sec]}", loc="left",
                     fontproperties=bold, color="0.1", pad=3)
        if i == 0:
            ax.text(0.5, 1.32, head, transform=ax.transAxes, ha="center",
                    fontproperties=ch3_style.bold_font_properties(size=10.5), color="0.1")
        if j == 0:
            ax.set_ylabel("10$^6$ km$^2$", fontsize=8, color=INK)
fig.tight_layout(h_pad=0.9, w_pad=1.5, rect=[0, 0, 1, 0.975])
fig.text(0.44, 0.995, "1979–2015", color=PRE, ha="right", va="top",
         fontproperties=ch3_style.bold_font_properties(size=9))
fig.text(0.56, 0.995, "2016–2025", color=POST, ha="left", va="top",
         fontproperties=ch3_style.bold_font_properties(size=9))
out = os.path.join(OUTPUT_DIR, "fig_s02_annual_min_max_trend.png")
fig.savefig(out, dpi=250, bbox_inches="tight")
plt.close(fig)

t = pd.DataFrame(rows)
t.to_csv(os.path.join(TABLES_DIR, "t_s02_annual_min_max_trend.csv"), index=False)
print(f"\nwrote {out}")
print(t.round(3).to_string(index=False))