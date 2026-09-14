"""
fig11_trend_volatility.py — the low-frequency term and the day-to-day volatility, per sector.
==============================================================================================
Two questions the annual scalars cannot answer, from the daily Pipeline-E file:

  (a) TREND: what shape is the GAM's low-frequency term s(t)? A step at 2016, a decline, or a
      reversal? Plotted as the annual mean of trend_component per sector, with the 2016 break.
  (b) VOLATILITY: has the day-to-day variability that the cycle model does not explain changed?
      Two measures: the annual SD of the APAC residual (Extent - fitted_apac), and the annual mean
      of the GARCH conditional SD ('volatility' column, fitted to that residual in 01_fit_apac.R).
      Both expressed as ratios post/pre-2016 with an F-test on the annual values.

Daily statistics start at YEAR_START (1980) because 1979's residual carries a spline edge effect
(sd ~1.5x later years).

Outputs
    results/ch3/figures/fig11_trend_volatility.png
    results/ch3/tables/t34b_trend_annual.csv               sector, Year, trend (annual mean of s(t))
    results/ch3/tables/t34b_volatility_annual.csv          sector, Year, sd_residual, mean_garch_sd
    results/ch3/tables/t34b_volatility_2016_ratio.csv      per sector: post/pre-2016 ratios + F-test p
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

import ch3_data as D
from ch3_config import (SECTORS, SECTOR_LABELS, SECTOR_COLORS, TABLES_DIR,
                        BREAK_YEAR, YEAR_START, YEAR_MAX)
from ch3_plot import mark_break, zero_line, year_axis, panel_letters, save

print("fig11 — trend term and volatility")
daily = D.load_daily()
D.summary(daily=daily)
d = daily[daily["Year"].between(YEAR_START, YEAR_MAX)].copy()

tr = (d.groupby(["sector", "Year"])["trend_component"].mean().reset_index()
        .rename(columns={"trend_component": "trend"}))
tr.to_csv(os.path.join(TABLES_DIR, "t34b_trend_annual.csv"), index=False)

vol = (d.groupby(["sector", "Year"])
         .agg(sd_residual=("residual_apac", lambda x: np.nanstd(x, ddof=1)),
              mean_garch_sd=("volatility", "mean"), n_days=("Date", "size"))
         .reset_index())
vol.to_csv(os.path.join(TABLES_DIR, "t34b_volatility_annual.csv"), index=False)

rows = []
for sec in SECTORS:
    g = vol[vol.sector == sec]; pre, post = g[g.Year < BREAK_YEAR], g[g.Year >= BREAK_YEAR]
    t = tr[tr.sector == sec]
    for col in ("sd_residual", "mean_garch_sd"):
        a, b = pre[col].dropna().values, post[col].dropna().values
        if len(b) < 3 or len(a) < 3: continue
        ratio = b.mean() / a.mean()
        # F on the annual values' variance is not the question; the question is level. Use Welch t on levels.
        tstat, p = stats.ttest_ind(b, a, equal_var=False)
        rows.append(dict(sector=SECTOR_LABELS[sec], measure=col, mean_pre=a.mean(), mean_post=b.mean(),
                         ratio_post_pre=ratio, p_welch=p, n_pre=len(a), n_post=len(b)))
    rows.append(dict(sector=SECTOR_LABELS[sec], measure="trend_mean", mean_pre=t[t.Year < BREAK_YEAR].trend.mean(),
                     mean_post=t[t.Year >= BREAK_YEAR].trend.mean(), ratio_post_pre=np.nan, p_welch=np.nan,
                     n_pre=int((t.Year < BREAK_YEAR).sum()), n_post=int((t.Year >= BREAK_YEAR).sum())))
summ = pd.DataFrame(rows)
summ.to_csv(os.path.join(TABLES_DIR, "t34b_volatility_2016_ratio.csv"), index=False)
print(summ.round(3).to_string(index=False))

# ── figure: 3 rows (trend, residual SD, GARCH SD) × 1, all sectors overlaid, circumpolar on its own axis
fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True, gridspec_kw=dict(width_ratios=[2.2, 1]))
panels = [("trend", tr, "trend", "s(t), annual mean (10⁶ km²)"),
          ("sd_residual", vol, "sd_residual", "SD of daily residual (10⁶ km²)"),
          ("mean_garch_sd", vol, "mean_garch_sd", "GARCH conditional SD, annual mean")]
for i, (key, df, col, ylab) in enumerate(panels):
    for j, secs in enumerate(([s for s in SECTORS if s != "SIE_circumpolar"], ["SIE_circumpolar"])):
        ax = axes[i, j]
        for sec in secs:
            g = df[df.sector == sec].sort_values("Year")
            ax.plot(g.Year, g[col], color=SECTOR_COLORS[sec], lw=1.8, label=SECTOR_LABELS[sec])
        if key == "trend": zero_line(ax)
        mark_break(ax); year_axis(ax, YEAR_START, YEAR_MAX)
        ax.spines[["top", "right"]].set_visible(False)
        if j == 0: ax.set_ylabel(ylab, fontsize=9)
        if i == 0: ax.set_title("sectors" if j == 0 else "circumpolar", fontsize=10, fontweight="bold")
axes[0, 0].legend(fontsize=8, frameon=False, ncol=3, loc="upper left")
panel_letters(axes.ravel())
fig.suptitle("Low-frequency term and unexplained day-to-day variability, by sector", fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
save(fig, "fig11_trend_volatility.png", sync=False)
