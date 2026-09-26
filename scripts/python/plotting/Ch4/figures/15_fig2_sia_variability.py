#!/usr/bin/env python
"""
15_fig2_sia_variability.py -- Ch4 Figure 2.
Question: did the day-to-day variability of sea-ice area change, and is it explained
by a smaller pack or by the wind?

Per sector (rows) x season (JJA, SON columns), 1988-2023, season-year variance of
  dSIA'        (solid, sector colour)
  dSIA'/SIA    (dashed, sector colour)  -- per unit of remaining ice
  wind stress' (thin grey)              -- the forcing
each divided by its own 1988-2015 mean (log axis; 1 = pre-2016 level).
Label = trend-adjusted 2016 step in var(dSIA') from log var ~ trend + step; * p < 0.05.
"""
import os
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ch4_style as st  # noqa: E402

ROOT = "/user/geog/falejandraperez/sea-ice-phase"
IN_CSV = f"{ROOT}/data/merged/analysis_table_daily_anomaly_clean.csv"
OUT = f"{ROOT}/results/ch4/figures/fig2_sia_variability.png"
START, BREAK = 1988, 2016
MIN_DAYS = 20
SEASONS = {"JJA": (6, 7, 8), "SON": (9, 10, 11)}
SECTORS = ["WS", "KH", "EA", "RA", "ABS"]
LONG = {"WS": "Weddell", "KH": "King Haakon VII", "EA": "East Antarctica",
        "RA": "Ross-Amundsen", "ABS": "Amundsen-Bellingshausen"}


def yearly(g, months):
    g = g[g.date.dt.month.isin(months)]
    y = g.groupby(g.date.dt.year).agg(raw=("delta_SIA_anomaly", "var"), rel=("rel", "var"),
                                      wind=("wind_stress_anomaly", "var"), n=("rel", "count"))
    return y[y.n >= MIN_DAYS]


def step_given_trend(t):
    yr = t.index.values.astype(float)
    X = np.column_stack([np.ones_like(yr), (yr - yr.mean()) / 10, (yr >= BREAK).astype(float)])
    m = sm.OLS(np.log(t.raw.values), X).fit()
    return (np.exp(m.params[2]) - 1) * 100, m.pvalues[2]


def main():
    df = pd.read_csv(IN_CSV, parse_dates=["date"])
    df = df[df.date.dt.year >= START].dropna(subset=["delta_SIA_anomaly", "SIA"])
    df = df[df.SIA > 0].copy()
    df["rel"] = df.delta_SIA_anomaly / df.SIA

    fig, axes = plt.subplots(len(SECTORS), 2, figsize=(7.2, 7.6), sharex=True, sharey=True)
    bold = st.bold_font_properties(size=9)
    for i, s in enumerate(SECTORS):
        g = df[df.sector == LONG[s]]
        col = st.SECTOR_COLORS[s]
        for j, (season, months) in enumerate(SEASONS.items()):
            ax = axes[i, j]
            t = yearly(g, months)
            pre = t.index < BREAK
            r = {k: t[k] / t.loc[pre, k].mean() for k in ("raw", "rel", "wind")}
            ax.axhline(1, color=st.GRID, lw=0.8, zorder=0)
            ax.axvline(BREAK - 0.5, color=st.INK, lw=0.6, ls=(0, (2, 2)), zorder=0)
            ax.plot(t.index, r["wind"], color="0.6", lw=0.9)
            ax.plot(t.index, r["rel"], color=col, lw=1.0, ls="--", alpha=0.8)
            ax.plot(t.index, r["raw"], color=col, lw=1.8)
            for mask in (pre, ~pre):     # period means (geometric), faint bars
                m = np.exp(np.log(r["raw"][mask]).mean())
                ax.hlines(m, t.index[mask].min(), t.index[mask].max(), color=col, lw=4, alpha=0.18)
            pct, p = step_given_trend(t)
            ax.text(0.99, 0.95, f"2016 step {pct:+.0f}%{'*' if p < 0.05 else ''}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=8, color=col if p < 0.05 else st.INK,
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.5, alpha=0.85))
            ax.set_yscale("log")
            ax.set_ylim(0.1, 5.5)
            ax.set_yticks([0.25, 0.5, 1, 2])
            ax.set_yticklabels(["¼", "½", "1", "2"])
            ax.minorticks_off()
            if i == 0:
                ax.set_title(season, loc="left")
            if j == 0:
                ax.text(0.01, 0.95, LONG[s].replace("-", "–"), transform=ax.transAxes,
                        ha="left", va="top", color=col, fontproperties=bold,
                        bbox=dict(facecolor="white", edgecolor="none", pad=0.5, alpha=0.85))
    # key as a single line of direct labels above the panels (no boxed legend)
    fig.text(0.08, 1.0, "——  var(ΔSIA′)", color=st.INK, fontsize=8, va="bottom")
    fig.text(0.30, 1.0, "- - -  var(ΔSIA′ / SIA), per unit of remaining ice", color=st.INK, fontsize=8, va="bottom")
    fig.text(0.78, 1.0, "——  var(wind stress′)", color="0.6", fontsize=8, va="bottom")
    fig.supylabel("Season-year variance relative to 1988–2015", fontsize=9, color=st.INK, x=0.0)
    fig.text(0.99, 0.005, "Label: 2016 step in var(ΔSIA′), trend-adjusted; * p < 0.05",
             ha="right", fontsize=7, color=st.INK)
    fig.tight_layout(rect=(0.02, 0.01, 1, 0.99), h_pad=0.6)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
