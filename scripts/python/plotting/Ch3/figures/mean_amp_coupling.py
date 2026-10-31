#!/usr/bin/env python
# make_fig_mean_amp_coupling.py -- Fig. 3.x: annual mean vs amplitude, before and after 2016.
# One panel per sector + circumpolar total. Each period is detrended within itself (as in the
# correlations of Sect. 3.3.2), so each cloud is centred on zero and the slope/r is what the text reports.
# Output: results/ch3/figures/fig_mean_amp_coupling.{png,pdf}
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt

from ch3_config import ROOT
ANNUAL = f"{ROOT}/data/ch3/annual_params.csv"
OUT = f"{ROOT}/results/ch3/figures"; os.makedirs(OUT, exist_ok=True)
SEC = [("Weddell", "SIE_Weddell"), ("King Haakon VII", "SIE_King_Haakon"),
       ("East Antarctica", "SIE_East_Antarctica"), ("Ross", "SIE_Ross"),
       ("Amundsen–Bellingshausen", "SIE_Amundsen_Bellingshausen"), ("Circumpolar", "SIE_circumpolar")]
PRE, POST = (1979, 2015), (2016, 2024)
C_PRE, C_POST = "0.65", "#D55E00"


def detr(s):
    x = s.index.values.astype(float)
    return s - np.polyval(np.polyfit(x, s.values, 1), x)


def main():
    a = pd.read_csv(ANNUAL); a = a[a.period == "FULL"] if "period" in a else a
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.4), constrained_layout=True)
    for ax, (title, key) in zip(axes.flat, SEC):
        g = a[a.sector == key].set_index("Year").sort_index()
        for (y0, y1), col, lab, z in ((PRE, C_PRE, "1979–2015", 1), (POST, C_POST, "2016–2024", 2)):
            d = g.loc[y0:y1, ["sie_annual", "amplitude_raw_yr"]].dropna().apply(detr)
            x, y = d.sie_annual.values, d.amplitude_raw_yr.values
            r = np.corrcoef(x, y)[0, 1]
            ax.scatter(x, y, s=22 if z == 1 else 34, color=col, edgecolor="white", linewidth=0.5, zorder=z,
                       label=f"{lab}  r = {r:+.2f}")
            xx = np.linspace(x.min(), x.max(), 10); b = np.polyfit(x, y, 1)
            ax.plot(xx, np.polyval(b, xx), color=col, lw=1.6, zorder=z)
            if z == 2:
                for yr, xi, yi in zip(d.index, x, y):
                    ax.annotate(str(yr)[2:], (xi, yi), xytext=(3, 2), textcoords="offset points",
                                fontsize=6.5, color=col)
        ax.axhline(0, color="0.85", lw=0.6, zorder=0); ax.axvline(0, color="0.85", lw=0.6, zorder=0)
        ax.set_title(title, fontsize=10); ax.legend(fontsize=7.5, frameon=False, loc="upper left")
        ax.tick_params(labelsize=8)
    for ax in axes[1]: ax.set_xlabel("Annual mean (10$^6$ km$^2$, detrended)", fontsize=9)
    for ax in axes[:, 0]: ax.set_ylabel("Amplitude (10$^6$ km$^2$, detrended)", fontsize=9)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/fig_mean_amp_coupling.{ext}", dpi=300)
    print(f"saved {OUT}/fig_mean_amp_coupling.png")


if __name__ == "__main__":
    main()