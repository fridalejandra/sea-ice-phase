"""
fig06_rolling_sd.py
===================
Rolling 10-year standard deviation of the observed metrics, by sector.
Asks whether year-to-year variability in timing and amplitude has changed.

Timing SDs are inflated by the flat-extremes problem (see fig04 caption), so
compare sectors and eras rather than reading absolute magnitudes as physical.
"""

import ch3_data as D
from ch3_config import SECTORS, SECTOR_COLORS, ROLL_SHORT
from ch3_plot import sector_grid, mark_break, year_axis, panel_letters, save

print("fig06 — rolling SD")
annual = D.load_annual()
D.summary(annual=annual)

METRICS = [
    ("min_doy_raw_anom",   "Minimum date",  "-",  1.0),
    ("max_doy_raw_anom",   "Maximum date",  "--", 0.65),
]

fig, axmap = sector_grid(2, 3, figsize=(15, 8))

for sec in SECTORS:
    ax = axmap[sec]; c = SECTOR_COLORS[sec]
    for col, lab, ls, alpha in METRICS:
        r = D.rolling_stat(annual, col, ROLL_SHORT, "std")
        g = r[r["sector"] == sec].sort_values("Year")
        ax.plot(g["Year"], g["value"], color=c, lw=2.0, ls=ls, alpha=alpha,
                label=f"{lab} (days)")

    ax2 = ax.twinx()
    ra = D.rolling_stat(annual, "amplitude_raw_anom", ROLL_SHORT, "std")
    ga = ra[ra["sector"] == sec].sort_values("Year")
    ax2.plot(ga["Year"], ga["value"], color="#2C2C2A", lw=1.6, alpha=0.55,
             label="Amplitude (Mkm²)")
    ax2.set_ylabel("Amplitude SD", fontsize=8, color="#2C2C2A")
    ax2.tick_params(axis="y", labelsize=8)
    ax2.spines[["top"]].set_visible(False)

    mark_break(ax); year_axis(ax)
    ax.set_ylabel(f"{ROLL_SHORT}-yr SD (days)", fontsize=9)

h1, l1 = axmap[SECTORS[0]].get_legend_handles_labels()
axmap[SECTORS[0]].legend(h1, l1, fontsize=7.5, loc="upper left", frameon=False)
panel_letters(list(axmap.values()))
fig.suptitle(f"Rolling {ROLL_SHORT}-year variability of the observed metrics",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.97])
save(fig, "fig06_rolling_sd.png")
