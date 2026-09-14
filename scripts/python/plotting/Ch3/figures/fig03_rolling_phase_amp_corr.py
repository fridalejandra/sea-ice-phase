"""
fig03_rolling_phase_amp_corr.py
===============================
Rolling correlation between timing and amplitude anomalies, by sector.

Tests whether phase and amplitude are independent, and whether that
independence has changed. Pre-2016 the observed correlations sit near zero in
every sector; post-2016 they diverge with sector-specific sign.

Statistic: TRAILING 10-year Spearman rho (plotted at the window's last year),
identical to t33_phase_amp_rolling10.csv from ch3_stats.py. The in-panel
numbers are Spearman rho for 1979-2015 and 2016-2023 (whole eras, not window
means). The shaded band is the p=0.05 threshold for the FULL record (n=45)
and is far too generous for a 10-year window; orientation only. The inference
in the text comes from the pooled whole-era split and leave-one-year-out
(ch3_numbers.md §3.3), not from this curve.
"""

import numpy as np
import ch3_data as D
from ch3_config import SECTORS, SECTOR_COLORS, ROLL_SHORT, BREAK_YEAR, R_SIG_45
from ch3_plot import sector_grid, mark_break, sig_band, year_axis, panel_letters, save

print("fig03 — rolling r(timing, amplitude)")
annual = D.load_annual()
D.summary(annual=annual)

roll_min = D.rolling_corr(annual, "min_doy_raw_anom", "amplitude_raw_anom", ROLL_SHORT, method="spearman")
roll_max = D.rolling_corr(annual, "max_doy_raw_anom", "amplitude_raw_anom", ROLL_SHORT, method="spearman")

fig, axmap = sector_grid(2, 3, figsize=(15, 8), sharey=True)

for sec in SECTORS:
    ax = axmap[sec]
    c = SECTOR_COLORS[sec]
    a = roll_min[roll_min["sector"] == sec].sort_values("Year")
    b = roll_max[roll_max["sector"] == sec].sort_values("Year")

    sig_band(ax, R_SIG_45)
    ax.plot(b["Year"], b["value"], color=c, lw=2.2, label="ρ(max date, amplitude)")
    ax.plot(a["Year"], a["value"], color=c, lw=1.6, ls="--", alpha=0.55,
            label="ρ(min date, amplitude)")

    mark_break(ax); year_axis(ax)
    ax.set_ylim(-1, 1)
    ax.set_ylabel(f"{ROLL_SHORT}-yr trailing Spearman ρ", fontsize=9)

    pre  = annual[(annual["sector"] == sec) & (annual["Year"] < BREAK_YEAR)]
    post = annual[(annual["sector"] == sec) & (annual["Year"] >= BREAK_YEAR)]
    r_pre  = pre["max_doy_raw_anom"].corr(pre["amplitude_raw_anom"], method="spearman")
    r_post = post["max_doy_raw_anom"].corr(post["amplitude_raw_anom"], method="spearman")
    ax.text(0.02, 0.04,
            f"max–amp (Spearman):  1979–2015 {r_pre:+.2f}   2016–2023 {r_post:+.2f}",
            transform=ax.transAxes, fontsize=7.5, color="#666666")

axmap[SECTORS[0]].legend(fontsize=7.5, loc="upper right", frameon=False)
panel_letters(list(axmap.values()))
fig.suptitle("Rolling correlation between observed timing and amplitude anomalies",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.97])
save(fig, "fig03_rolling_phase_amp_corr.png")