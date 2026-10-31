#!/usr/bin/env python
"""ch3_fig05_mean_amp_coupling.py -- Fig. 3.5: amplitude against annual mean,
1979-2015 (grey) and 2016-2024 (vermillion, labelled by year), one panel per sector
plus the circumpolar total. Each period is detrended within itself, as in Sect. 3.3.2,
so the r in each legend is the one quoted in the text."""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ch3_style as S
from ch3_config import ANNUAL_CSV

ORDER = ["SIE_Weddell", "SIE_King_Haakon", "SIE_East_Antarctica",
         "SIE_Ross", "SIE_Amundsen_Bellingshausen", "SIE_circumpolar"]
PRE, POST = (1979, 2015), (2016, 2024)


def detr(s):
    x = s.index.values.astype(float)
    return s - np.polyval(np.polyfit(x, s.values, 1), x)


def main():
    a = pd.read_csv(ANNUAL_CSV)
    if "period" in a:
        a = a[a.period == "FULL"]
    fig, axes = plt.subplots(2, 3, figsize=(S.TEXT_WIDTH_IN, 4.6), constrained_layout=True)
    for ax, key in zip(axes.flat, ORDER):
        g = a[a.sector == key].set_index("Year").sort_index()
        for (y0, y1), col, lab, z in ((PRE, S.GREY, "1979–2015", 1), (POST, S.VERMILLION, "2016–2024", 2)):
            d = g.loc[y0:y1, ["sie_annual", "amplitude_raw_yr"]].dropna().apply(detr)
            x, y = d.sie_annual.values, d.amplitude_raw_yr.values
            r = np.corrcoef(x, y)[0, 1]
            ax.scatter(x, y, s=12 if z == 1 else 20, color=col, edgecolor="white", linewidth=0.4,
                       zorder=z, label=lab)
            ax.text(0.98, 0.04 + 0.09 * (z - 1), f"r = {r:+.2f}", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=7, color=col)
            xx = np.linspace(x.min(), x.max(), 10)
            ax.plot(xx, np.polyval(np.polyfit(x, y, 1), xx), color=col, lw=1.3, zorder=z)
            if z == 2:
                for yr, xi, yi in zip(d.index, x, y):
                    ax.annotate(str(yr)[2:], (xi, yi), xytext=(2, 2), textcoords="offset points",
                                fontsize=5, color=col)
            print(f"{S.SECTOR_NAMES[key]:24s} {lab}: r = {r:+.2f}")
        ax.axhline(0, color=S.GREY2, lw=0.6, zorder=0); ax.axvline(0, color=S.GREY2, lw=0.6, zorder=0)
        ax.set_title(S.SECTOR_NAMES[key], fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("Annual mean (10$^6$ km$^2$)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Amplitude (10$^6$ km$^2$)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04), fontsize=7.5)
    S.label_panels(axes)
    S.save_figure(fig, "ch3_fig05_mean_amp_coupling")


if __name__ == "__main__":
    main()
