"""
fig05_amplitude_timeseries.py
=============================
Observed amplitude anomalies by sector, 1980–2023.

Amplitude = observed annual maximum minus observed annual minimum, so it is
measured on the same footing as the timing metrics in fig04. No fitted
quantities are used.
"""

import numpy as np
import ch3_data as D
from ch3_config import SECTORS, SECTOR_COLORS
from ch3_plot import sector_grid, mark_break, zero_line, year_axis, panel_letters, save

print("fig05 — observed amplitude anomalies")
annual = D.load_annual()
D.summary(annual=annual)

fig, axmap = sector_grid(2, 3, figsize=(15, 8))

for sec in SECTORS:
    ax = axmap[sec]
    g = annual[annual["sector"] == sec].sort_values("Year")
    c = SECTOR_COLORS[sec]

    vals = g["amplitude_raw_anom"].values
    ax.bar(g["Year"], vals, color=c, alpha=0.75, width=0.75,
           edgecolor="white", linewidth=0.4)
    ax.plot(g["Year"], g["amplitude_raw_anom"].rolling(5, center=True).mean(),
            color="#2C2C2A", lw=1.6, alpha=0.8, label="5-yr mean")

    zero_line(ax); mark_break(ax); year_axis(ax)
    ax.set_ylabel("Amplitude anomaly\n(million km²)", fontsize=9)

    # flag the most extreme year
    i = int(np.nanargmax(np.abs(vals)))
    ax.annotate(f"{int(g['Year'].iloc[i])}", xy=(g["Year"].iloc[i], vals[i]),
                xytext=(0, 8 if vals[i] < 0 else -14), textcoords="offset points",
                fontsize=7.5, ha="center", color="#2C2C2A")

axmap[SECTORS[0]].legend(fontsize=8, loc="upper right", frameon=False)
panel_letters(list(axmap.values()))
fig.suptitle("Observed amplitude anomalies of the annual cycle, by sector",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.97])
save(fig, "fig05_amplitude_timeseries.png")
