#!/usr/bin/env python3
"""
plot_variance_by_year.py -- backup slide: year-by-year variance of the daily area tendency.

One panel, five sectors. Each year's variance of dSIA (day-to-day change in sector sea-ice
area) is divided by that sector's 1992-2015 mean, so the sectors share an axis; log scale.
SMMR years (every-other-day sampling) shaded; 1988 (SSM/I changeover and the Dec 1987-Jan
1988 gap) and 2016 marked.  Transparent background, Helvetica Neue if available.

Run (seaice env):
  python plot_variance_by_year.py
  python plot_variance_by_year.py --sia /path/to/sia_by_sector_daily.csv --out variance_by_year.png
"""
import argparse
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

SIA = "/user/geog/falejandraperez/sea-ice-phase/data/merged/sia_by_sector_daily.csv"
COLORS = {"Weddell": "#D55E00", "King Haakon": "#E69F00", "East Antarctica": "#009E73",
          "Ross": "#0072B2", "Amundsen": "#CC79A7"}
INK, MUTED = "#222222", "#6B6B6B"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sia", default=SIA)
    p.add_argument("--out", default="variance_by_year.png")
    p.add_argument("--first", type=int, default=1979)
    p.add_argument("--last", type=int, default=2024)
    a = p.parse_args()

    fam = "Helvetica Neue" if any("Helvetica Neue" in f.name for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
    plt.rcParams.update({"font.family": fam, "font.size": 13, "axes.edgecolor": "grey",
                         "xtick.color": INK, "ytick.color": INK, "axes.labelcolor": MUTED})

    d = pd.read_csv(a.sia)
    dc = [c for c in d.columns if c.lower() in ("date", "time")][0]
    d[dc] = pd.to_datetime(d[dc]); d = d.set_index(dc).sort_index()
    cols = [c for c in d.columns if not c.endswith("coverage") and "circum" not in c.lower()
            and d[c].dtype.kind == "f"]

    fig, ax = plt.subplots(figsize=(13, 5.6))
    fig.patch.set_alpha(0); ax.set_facecolor("none")
    ax.axvspan(a.first - 0.5, 1987.5, color="#000000", alpha=0.06, lw=0)
    for yr, lab in [(1988, "1988: SSM/I, data gap"), (2016, "2016")]:
        ax.axvline(yr, color="grey", ls=":", lw=1.4)
        ax.text(yr + 0.3, 0.93, lab, transform=ax.get_xaxis_transform(), color=MUTED, fontsize=11, va="top")
    ax.text((a.first + 1987) / 2, 0.93, "SMMR: every other day", transform=ax.get_xaxis_transform(),
            color=MUTED, fontsize=11, ha="center", va="top")

    for c in cols:
        v = d[c].diff().groupby(d.index.year).var()
        v = v[(v.index >= a.first) & (v.index <= a.last)]
        base = v[(v.index >= 1992) & (v.index <= 2015)].mean()
        key = next((k for k in COLORS if k.lower() in c.lower()), None)
        col = COLORS.get(key, "grey")
        ax.plot(v.index, v / base, color=col, lw=2.4, solid_capstyle="round", label=c.replace("_", " "))

    ax.axhline(1, color=INK, lw=1, ls="--")
    leg = ax.legend(loc="upper right", bbox_to_anchor=(0.98, 0.84), frameon=False, ncol=1, fontsize=12,
                    handlelength=1.6, labelcolor="linecolor")
    for t in leg.get_texts(): t.set_fontweight("bold")
    ax.set_yscale("log")
    ax.set_yticks([0.5, 1, 2, 4, 8, 16]); ax.set_yticklabels(["0.5×", "1×", "2×", "4×", "8×", "16×"])
    ax.set_ylabel("Variance of daily area tendency\n(relative to 1992–2015 mean)")
    ax.set_xlim(a.first - 0.5, a.last + 0.5)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.tick_params(length=4, color="grey")
    fig.tight_layout()
    fig.savefig(a.out, dpi=300, transparent=True)
    fig.savefig(a.out.rsplit(".", 1)[0] + ".svg", transparent=True)
    print("wrote", a.out)

if __name__ == "__main__":
    main()
