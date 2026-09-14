"""
figS01_fitted_vs_raw.py
=======================
Why the chapter uses OBSERVED extremum dates and not fitted-curve peaks.

The argmax/argmin of the fitted APAC curve is dominated by the fixed s(DOY)
term, so it barely moves between years: its SD is several times smaller than
the observed, and it correlates weakly (or, in Ross, NEGATIVELY with the
observed MINIMUM). It is not a timing metric. This figure is the evidence.
"""

import numpy as np
import matplotlib.pyplot as plt

import ch3_data as D
from ch3_config import SECTORS, SECTOR_LABELS, SECTOR_COLORS
from ch3_plot import panel_letters, save

print("figS01 — fitted vs observed timing")
annual = D.load_annual()
D.summary(annual=annual)

fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))

for ax, sec in zip(axes.ravel(), SECTORS):
    g = annual[annual["sector"] == sec]
    c = SECTOR_COLORS[sec]
    ax.scatter(g["max_doy_raw"], g["max_doy_fitted"], s=28, color=c,
               alpha=0.75, edgecolors="white", linewidth=0.5)

    lims = [min(g["max_doy_raw"].min(), g["max_doy_fitted"].min())-8,
            max(g["max_doy_raw"].max(), g["max_doy_fitted"].max())+8]
    ax.plot(lims, lims, color="#888888", ls="--", lw=1.0, label="1:1")
    ax.set_xlim(lims); ax.set_ylim(lims)

    r_max = g["max_doy_fitted"].corr(g["max_doy_raw"])
    r_min = g["max_doy_fitted"].corr(g["min_doy_raw"])
    sd_f, sd_r = g["max_doy_fitted"].std(), g["max_doy_raw"].std()
    ax.text(0.03, 0.97,
            f"r(fitted max, raw max) = {r_max:+.2f}\n"
            f"r(fitted max, raw MIN) = {r_min:+.2f}\n"
            f"SD fitted {sd_f:.1f} d  vs raw {sd_r:.1f} d",
            transform=ax.transAxes, fontsize=7.5, va="top",
            bbox=dict(fc="white", ec="#DDDDDD", alpha=0.85, pad=3))

    ax.set_title(SECTOR_LABELS[sec], fontsize=11, fontweight="bold", color=c)
    ax.set_xlabel("Observed maximum DOY", fontsize=9)
    ax.set_ylabel("Fitted-curve maximum DOY", fontsize=9)
    ax.spines[["top","right"]].set_visible(False)

axes.ravel()[0].legend(fontsize=8, loc="lower right", frameon=False)
panel_letters(axes)
fig.suptitle("Fitted-curve peak is not a timing metric: it is compressed "
             "relative to the observed maximum", fontsize=12, fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.95])
save(fig, "figS01_fitted_vs_raw.png")
