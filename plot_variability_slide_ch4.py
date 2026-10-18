#!/usr/bin/env python3
"""
plot_variability_slide.py -- supplementary slide from results/ch4_variability_cd.csv.

Left: percent change in the variance of deseasonalized daily anomalies, 2016-2024 vs 1992-2015.
Right: percent change in the mean (raw daily series).  One row per quantity, one dot per sector
with its 95 % year-block-bootstrap interval; filled dots where the interval excludes zero.
dSIA has no meaningful mean change (its mean is ~0), so it is shown on the left only.

Run (seaice env):
  python plot_variability_slide.py
  python plot_variability_slide.py --csv results/ch4_variability_cd.csv --out variability_slide.png
"""
import argparse
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

CSV = "/user/geog/falejandraperez/sea-ice-phase/results/ch4_variability_cd.csv"
ROWS = [("dSIA", "area tendency"), ("wind", "wind speed"), ("drift", "drift, NSIDC-0116"),
        ("drift_osi", "drift, OSI-455"), ("div", "divergence"), ("conv", "convergence")]
SECT = [("WED", "Weddell", "#D55E00"), ("KHV", "King Haakon", "#E69F00"), ("EA", "East Antarctica", "#009E73"),
        ("RA", "Ross–Amundsen", "#0072B2"), ("ABS", "Amundsen–Bellingshausen", "#CC79A7")]
INK, MUTED = "#222222", "#6B6B6B"

def panel(ax, d, suffix, title, xlim, skip=()):
    ax.set_facecolor("none")
    ax.axvline(0, color=INK, lw=1, ls="--")
    for i, (q, lab) in enumerate(ROWS):
        y = len(ROWS) - 1 - i
        if q in skip:
            ax.text(0, y, "mean ≈ 0 by construction", ha="center", va="center", color=MUTED, fontsize=10); continue
        g = d[d["quantity"] == q + suffix]
        for j, (s, _, col) in enumerate(SECT):
            r = g[g["sector"] == s]
            if r.empty: continue
            r = r.iloc[0]
            yy = y + (j - 2) * 0.13
            sig = (r.ci_lo > 0) or (r.ci_hi < 0)
            ax.plot([r.ci_lo, r.ci_hi], [yy, yy], color=col, lw=1.6, alpha=0.8, solid_capstyle="round")
            ax.plot(r.change_pct, yy, "o", ms=7, mfc=col if sig else "white", mec=col, mew=1.8, zorder=3)
    ax.set_yticks(range(len(ROWS))); ax.set_yticklabels([lab for _, lab in ROWS][::-1], fontsize=13)
    ax.set_xlim(*xlim); ax.set_ylim(-0.6, len(ROWS) - 0.4)
    ax.set_xlabel(title, color=MUTED)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.0f}%"))
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0); ax.tick_params(axis="x", color="grey", length=4)
    ax.grid(axis="x", color="#000000", alpha=0.08, lw=0.8)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=CSV)
    p.add_argument("--out", default="variability_slide.png")
    a = p.parse_args()
    fam = "Helvetica Neue" if any("Helvetica Neue" in f.name for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
    plt.rcParams.update({"font.family": fam, "font.size": 12, "axes.edgecolor": "grey",
                         "xtick.color": INK, "ytick.color": INK, "axes.labelcolor": MUTED})
    d = pd.read_csv(a.csv)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), sharey=False)
    fig.patch.set_alpha(0)
    fig.subplots_adjust(left=0.17, right=0.98, top=0.9, bottom=0.2, wspace=0.18)
    panel(ax1, d, "", "Change in day-to-day variance, 2016–24 vs 1992–2015", (-40, 40))
    panel(ax2, d, "_mean", "Change in the mean, 2016–24 vs 1992–2015", (-15, 15), skip=("dSIA",))
    ax2.set_yticklabels([])
    handles = [plt.Line2D([], [], marker="o", ls="", mfc="white", mec=c, mew=1.8, ms=7, label=n) for _, n, c in SECT]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=11, bbox_to_anchor=(0.57, 0.01))
    fig.text(0.57, 0.965, "Filled dot: 95 % interval excludes zero.  Bootstrap over years.  Divergence and convergence: NSIDC-0116 drift at 25 km, sector mean over ice.",
             ha="center", va="top", color=MUTED, fontsize=10)
    fig.savefig(a.out, dpi=300, transparent=True)
    fig.savefig(a.out.rsplit(".", 1)[0] + ".svg", transparent=True)
    print("wrote", a.out)

if __name__ == "__main__":
    main()
