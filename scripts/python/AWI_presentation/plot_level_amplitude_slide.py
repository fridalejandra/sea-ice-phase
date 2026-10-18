#!/usr/bin/env python3
"""
plot_level_amplitude_slide.py -- slide version of Ch. 3 Fig. 3.8(a).

The circumpolar LEVEL (annual-mean extent) and AMPLITUDE (June-Nov maximum minus Jan-Mar minimum),
both linearly detrended over 1979-END, in 10^6 km^2. Same definitions and detrending as
fig_level_vs_amplitude.py, so the numbers match the chapter.

Changes for the talk:
  - standalone (no ch3_config / ch3_style); reads daily_fitted.csv directly
  - panel (b), legend, notes box and titles removed; the annotations are added in PowerPoint
  - level and amplitude in TWO stacked rows on the SAME y-axis, so overlap does not hide which one moved
  - colours match the anatomy slide (level green, amplitude amber)
  - transparent background; SHOW_LABELS = True draws the numbers inside the figure instead

Usage:
    python plot_level_amplitude_slide.py [END]            # default END = 2025
    python plot_level_amplitude_slide.py 2023             # record ended in 2023

Writes {OUT_DIR}/level_amplitude_slide.png (300 dpi, transparent) and .svg, and prints the numbers
to put on the slide (variance ratios, sub-period level means).
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# ---- paths ------------------------------------------------------------------
ROOT = "/Users/fridaperez/Research/repos/sea-ice-phase"
DAILY = os.path.join(ROOT, "data", "ch3", "daily_fitted.csv")
OUT_DIR = os.path.join(ROOT, "results", "ch3", "figures")
SECTOR = "SIE_circumpolar"
PERIOD = "FULL"

# ---- definitions (as in the chapter) ----------------------------------------
YEAR_MIN = 1979
END = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
SPLIT = 2007          # variance before / after
BREAK_YEAR = 2016     # sub-period means: SPLIT..BREAK_YEAR-1 and BREAK_YEAR..END

# ---- look ---------------------------------------------------------------------
INK = "#222222"       # use "#EEEEEE" on a dark slide
MUTED = "#6B6B6B"
COL_LEVEL = "#009E73"   # same as trend / level on the anatomy slide
COL_AMP = "#E69F00"     # same as amplitude on the anatomy slide
FONT = None             # e.g. "Arial" or "Helvetica"; match the other slide figures
SHOW_LABELS = False     # True: draw the row names and numbers in the figure
SHADE_LAST = True       # faint band over 2024-25 (noisy record); no text
W, H = 11.0, 5.6


def detrend(s):
    t = s.index.values.astype(float)
    m, b = np.polyfit(t, s.values, 1)
    return s - (m * t + b)


def main():
    if FONT:
        matplotlib.rcParams["font.family"] = FONT
    matplotlib.rcParams.update({"text.color": INK, "axes.labelcolor": INK,
                                "xtick.color": INK, "ytick.color": INK})
    if not os.path.exists(DAILY):
        sys.exit(f"Input not found: {DAILY}")
    os.makedirs(OUT_DIR, exist_ok=True)

    d = pd.read_csv(DAILY, parse_dates=["Date"])
    if "period" in d.columns:
        d = d[d["period"] == PERIOD]
    g = d[d["sector"] == SECTOR].copy()
    if g.empty:
        sys.exit(f"no rows for sector {SECTOR}; have: {sorted(d['sector'].unique())}")
    col = next(c for c in ("Extent", "extent", "SIE", "sie") if c in g.columns)
    if g[col].max() > 1000:                       # km^2 -> 10^6 km^2
        g[col] = g[col] * 1e-6
    g["Y"], g["m"] = g["Date"].dt.year, g["Date"].dt.month
    g = g[g["Y"].between(YEAR_MIN, END)]

    t = pd.DataFrame({
        "min": g[g.m.between(1, 3)].groupby("Y")[col].min(),
        "max": g[g.m.between(6, 11)].groupby("Y")[col].max(),
        "level": g.groupby("Y")[col].mean()}).dropna()
    t["amp"] = t["max"] - t["min"]
    c = t[["level", "amp"]].apply(detrend)

    def vratio(s):
        return s[s.index >= SPLIT].var(ddof=1) / s[s.index < SPLIT].var(ddof=1)

    r_lev, r_amp = vratio(c["level"]), vratio(c["amp"])
    m1 = c["level"][(c.index >= SPLIT) & (c.index < BREAK_YEAR)].mean()
    m2 = c["level"][c.index >= BREAK_YEAR].mean()
    print(f"variance after/before {SPLIT}: level x{r_lev:.2f}, amplitude x{r_amp:.2f}")
    print(f"level means: {SPLIT}-{BREAK_YEAR - 1} {m1:+.2f}, {BREAK_YEAR}-{END} {m2:+.2f} (10^6 km^2, detrended)")
    print("(chapter: x3.9 / x0.8 and +0.65 / -0.49 with END = 2025)")

    # one shared y-range, so the rows can be compared honestly
    ymax = max(c["level"].abs().max(), c["amp"].abs().max()) * 1.12
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(W, H), sharex=True, sharey=True)
    fig.patch.set_alpha(0)

    for ax in (ax1, ax2):
        ax.set_facecolor("none")
        ax.axhline(0, color="0.75", lw=1.0, zorder=0)
        ax.axvline(SPLIT - 0.5, color="0.55", lw=1.4, ls=(0, (2, 2)), zorder=1)
        if SHADE_LAST and END >= 2024:
            ax.axvspan(2023.5, END + 0.5, color="0.5", alpha=0.12, lw=0, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color("0.6")
        ax.spines["bottom"].set_color("0.6")
        ax.tick_params(labelsize=15, length=4)
        ax.set_ylim(-ymax, ymax)

    # level, with each sub-period mean
    ax1.plot(c.index, c["level"], color=COL_LEVEL, lw=3.6, solid_capstyle="round", zorder=3)
    for y0, y1, m in ((SPLIT, BREAK_YEAR - 1, m1), (BREAK_YEAR, END, m2)):
        ax1.plot([y0 - 0.4, y1 + 0.4], [m, m], color=INK, lw=1.6, ls=(0, (4, 2)), zorder=4)
        if SHOW_LABELS:
            ax1.text(y1 + 0.7, m, f"{m:+.2f}".replace("-", "−"), ha="left", va="center",
                     fontsize=15, color=INK)
    # amplitude
    ax2.plot(c.index, c["amp"], color=COL_AMP, lw=3.6, solid_capstyle="round", zorder=3)

    if SHOW_LABELS:
        ax1.text(0.01, 0.93, f"level (annual mean):  variance after {SPLIT} ×{r_lev:.1f}",
                 transform=ax1.transAxes, ha="left", va="top", fontsize=16,
                 fontweight="bold", color=COL_LEVEL)
        ax2.text(0.01, 0.93, f"amplitude (max − min):  variance after {SPLIT} ×{r_amp:.1f}",
                 transform=ax2.transAxes, ha="left", va="top", fontsize=16,
                 fontweight="bold", color=COL_AMP)

    ax2.set_xlim(YEAR_MIN - 0.5, END + 0.5)
    ax2.set_xticks([y for y in range(1980, END + 1, 10)])
    fig.supylabel("Detrended anomaly (10$^6$ km$^2$)", fontsize=16, color=MUTED, x=0.01)
    fig.tight_layout(h_pad=1.0)

    stem = os.path.join(OUT_DIR, "level_amplitude_slide" + ("" if END == 2025 else f"_to{END}"))
    fig.savefig(stem + ".svg", transparent=True)
    fig.savefig(stem + ".png", dpi=300, transparent=True)
    print("wrote", stem + ".png / .svg")


if __name__ == "__main__":
    main()
