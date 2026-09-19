"""
fig04_phase_timeseries.py
=========================
Observed timing anomalies by sector, 1980–2023.

Plots min-date and max-date anomalies from annual_params_B.csv. These are
OBSERVED extremum dates, not fitted-curve peaks: the argmax of the fitted
curve is dominated by the fixed s(DOY) term and is an artifact (see S01).

Limitation shown by the metric itself: the annual cycle is flat near its
extremes, so noise displaces the recorded argmin/argmax by days to weeks.
The climatological curve stays within 1% of its minimum for 9–33 days
depending on sector, while daily weather noise is 0.03–0.17 million km².
Resulting SDs are 4–17 days (min) and 11–22 days (max). Weddell is worst.
"""

import numpy as np
import matplotlib.pyplot as plt

import ch3_data as D
from ch3_config import SECTORS, SECTOR_COLORS, SECTOR_LABELS, BREAK_YEAR
from ch3_plot import sector_grid, mark_break, zero_line, year_axis, panel_letters, save

print("fig04 — observed timing anomalies")
annual = D.load_annual()
D.summary(annual=annual)

fig, axmap = sector_grid(2, 3, figsize=(15, 8))

for sec in SECTORS:
    ax = axmap[sec]
    g = annual[annual["sector"] == sec].sort_values("Year")
    c = SECTOR_COLORS[sec]

    ax.plot(g["Year"], g["min_doy_raw_anom"], color=c, lw=1.4,
            marker="o", ms=3.5, label="Minimum date")
    ax.plot(g["Year"], g["max_doy_raw_anom"], color=c, lw=1.4, ls="--",
            marker="s", ms=3.5, alpha=0.65, label="Maximum date")

    zero_line(ax)
    mark_break(ax)
    year_axis(ax)
    ax.set_ylabel("Anomaly (days)\n← earlier | later →", fontsize=9)

    sd_min = g["min_doy_raw_anom"].std()
    sd_max = g["max_doy_raw_anom"].std()
    ax.text(0.02, 0.04, f"SD  min {sd_min:.1f} d   max {sd_max:.1f} d",
            transform=ax.transAxes, fontsize=7.5, color="#666666")

axmap[SECTORS[0]].legend(fontsize=8, loc="upper right", frameon=False)
panel_letters(list(axmap.values()))
fig.suptitle("Observed timing anomalies of the annual cycle, by sector",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.97])
save(fig, "fig04_phase_timeseries.png")
