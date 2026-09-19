"""
fig12_volatility_seasonal.py — seasonal cycle of day-to-day volatility, 1988-2015 vs 2016-2023.
Reads results/ch3/tables/t34c_volatility_seasonal_curves.csv (from R/ch3/05_volatility_gamlss.R):
fitted sigma of the daily tendency by day of year, per sector, separate fits by period, SSMIS sensor level.
"""
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from ch3_config import SECTORS, SECTOR_LABELS, SECTOR_COLORS, TABLES_DIR, BREAK_YEAR
from ch3_plot import sector_grid, panel_letters, save

cur = pd.read_csv(os.path.join(TABLES_DIR, "t34c_volatility_seasonal_curves.csv"))
cur = cur[cur.response == "dSIE"]
fig, axmap = sector_grid(2, 3, sharex=True, sharey=False)
for sec in SECTORS:
    ax = axmap[sec]; g = cur[cur.sector == sec].sort_values("DOY"); c = SECTOR_COLORS[sec]
    ax.plot(g.DOY, g.sigma_pre, color=c, lw=2.2, label=f"1988–{BREAK_YEAR-1}")
    ax.plot(g.DOY, g.sigma_post, color=c, lw=2.0, ls="--", label=f"{BREAK_YEAR}–2023")
    ax.set_xlim(1, 365); ax.set_xticks([1, 60, 121, 182, 244, 305]); ax.set_xticklabels(["Jan", "Mar", "May", "Jul", "Sep", "Nov"])
    ax.set_ylabel("σ of daily tendency (10⁶ km² d⁻¹)", fontsize=9)
    ax.text(0.02, 0.04, f"post/pre = {g.sigma_post.mean()/g.sigma_pre.mean():.2f}", transform=ax.transAxes, fontsize=8, color="#555555")
axmap[SECTORS[0]].legend(fontsize=8, frameon=False, loc="upper right")
panel_letters(list(axmap.values()))
fig.suptitle("Seasonal cycle of day-to-day volatility, before and after 2016 (gamlss, sensor held fixed)", fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
save(fig, "fig12_volatility_seasonal.png", sync=False)