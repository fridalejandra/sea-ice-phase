#!/usr/bin/env python3
"""
fig_level_vs_amplitude.py -- Sect. 3.4: the recent increase in variability is in the LEVEL of the
cycle, not its size, and it comes from the sectors moving together.

(a) Circumpolar total: the level (annual-mean extent) and the amplitude (June-November maximum minus
    January-March minimum), both linearly detrended over 1979-END, on one axis in 10^6 km^2. The level
    swings far more after 2007 -- high in 2007-2015, low from 2016 (dashed: each sub-period's mean) --
    while the amplitude does not.
(b) The five sectors' levels, detrended the same way: before 2007 they partly offset one another,
    after 2007 they rise and fall together.

Same definitions and detrending as checks/check_level_variance.py and checks/check_level_synchrony.py,
so the numbers printed in the panels match Sect. 3.4. 2024-2025 are shaded: their day-to-day variance
jumps in every sector at once (possible change in the satellite record; see Sect. 3.4).

Reads   DAILY_CSV (via ch3_config)
Writes  results/ch3/figures/fig_level_vs_amplitude.png              (white bg, manuscript)
        results/ch3/figures/fig_level_vs_amplitude_presentation.png (transparent bg, slides)
Run     python fig_level_vs_amplitude.py [END]      (default END = 2025)
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ch3_config import DAILY_CSV, OUTPUT_DIR, TEXT_WIDTH_IN, SECTORS_ONLY, SECTOR_LABELS, SECTOR_COLORS, YEAR_MIN, BREAK_YEAR
import ch3_style

END = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
SPLIT = 2007
INK, LEVEL_C, AMP_C = ch3_style.INK, "0.12", "0.6"
NAMES = {"SIE_Weddell": "Weddell", "SIE_Amundsen_Bellingshausen": "Amundsen–Bellingshausen", "SIE_Ross": "Ross",
         "SIE_East_Antarctica": "East Antarctica", "SIE_King_Haakon": "King Haakon"}


def detrend(s):
    t = s.index.values.astype(float); m, b = np.polyfit(t, s.values, 1)
    return s - (m * t + b)


d = pd.read_csv(DAILY_CSV, parse_dates=["Date"])
if "period" in d.columns:
    d = d[d["period"] == "FULL"]
col = next(c for c in ("Extent", "extent", "SIE", "sie") if c in d.columns)
d["Y"], d["m"] = d["Date"].dt.year, d["Date"].dt.month
d = d[d["Y"].between(YEAR_MIN, END)]


def cycle_stats(sec):
    g = d[d["sector"] == sec]
    t = pd.DataFrame({"min": g[g.m.between(1, 3)].groupby("Y")[col].min(),
                      "max": g[g.m.between(6, 11)].groupby("Y")[col].max(),
                      "level": g.groupby("Y")[col].mean()}).dropna()
    t["amp"] = t["max"] - t["min"]
    return t[["level", "amp"]].apply(detrend)


circ = cycle_stats("SIE_circumpolar")
lev = pd.DataFrame({s: cycle_stats(s)["level"] for s in SECTORS_ONLY}).dropna()


def vratio(s):
    return s[s.index >= SPLIT].var(ddof=1) / s[s.index < SPLIT].var(ddof=1)


r_lev, r_amp = vratio(circ["level"]), vratio(circ["amp"])
m1 = circ["level"][(circ.index >= SPLIT) & (circ.index < BREAK_YEAR)].mean()
m2 = circ["level"][circ.index >= BREAK_YEAR].mean()


def mean_pair_r(df):
    c = df.corr().values; return c[np.triu_indices_from(c, 1)].mean()


rc_pre, rc_post = mean_pair_r(lev[lev.index < SPLIT]), mean_pair_r(lev[lev.index >= SPLIT])
print(f"circumpolar variance after/before {SPLIT}: level x{r_lev:.2f}, amplitude x{r_amp:.2f}")
print(f"level means: {SPLIT}-{BREAK_YEAR - 1} {m1:+.2f}, {BREAK_YEAR}-{END} {m2:+.2f} (10^6 km^2, detrended)")
print(f"mean pairwise correlation of sector levels: {rc_pre:+.2f} before, {rc_post:+.2f} after")

# ---- figure -----------------------------------------------------------------
title_fp = ch3_style.bold_font_properties(size=10)
fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(TEXT_WIDTH_IN, 5.6), sharex=True)

for ax in (ax_a, ax_b):
    ax.axhline(0, color="0.8", lw=0.8, zorder=0)
    ax.axvline(SPLIT - 0.5, color="0.55", lw=0.9, ls=(0, (2, 2)), zorder=0)
    if END >= 2024:
        ax.axvspan(2023.5, END + 0.5, color="0.93", zorder=-1, lw=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8, colors=INK)
    ax.set_ylabel("Detrended anomaly (10$^6$ km$^2$)", fontsize=8.5, color=INK)

# (a) circumpolar level and amplitude
ax_a.plot(circ.index, circ["amp"], color=AMP_C, lw=1.6, label="amplitude (maximum − minimum)", zorder=2)
ax_a.plot(circ.index, circ["level"], color=LEVEL_C, lw=2.0, label="level (annual mean)", zorder=3)
for (y0, y1, m) in ((SPLIT, BREAK_YEAR - 1, m1), (BREAK_YEAR, END, m2)):
    ax_a.plot([y0 - 0.3, y1 + 0.3], [m, m], color=LEVEL_C, lw=1.0, ls=(0, (4, 2)), zorder=4)
    ax_a.text(y1 + 0.6, m, f"{m:+.2f}".replace("-", "\u2212"), ha="left", va="center", fontsize=8,
              color=LEVEL_C, bbox=dict(facecolor="white", edgecolor="none", pad=0.8), zorder=5)
ax_a.text(0.01, 0.97, f"variance after {SPLIT} vs before:  level ×{r_lev:.1f},  amplitude ×{r_amp:.1f}",
          transform=ax_a.transAxes, ha="left", va="top", fontsize=8, color="0.2", bbox=dict(facecolor="white", edgecolor="none", pad=1.5), zorder=6)
ax_a.legend(frameon=False, fontsize=8, loc="lower left", ncol=2, handlelength=1.8)
ax_a.set_title("(a) Circumpolar total: level and size of the cycle", loc="left", fontproperties=title_fp,
               color="0.1", pad=6)
if END >= 2024:
    ax_a.text(2024, ax_a.get_ylim()[1], "2024–25", ha="center", va="top", fontsize=7, color="0.5")

# (b) sector levels
for s in SECTORS_ONLY:
    ax_b.plot(lev.index, lev[s], color=SECTOR_COLORS[s], lw=1.4, label=NAMES.get(s, SECTOR_LABELS.get(s, s)))
ax_b.text(0.01, 0.97, f"mean correlation between sector levels:  {rc_pre:+.2f} before {SPLIT},  {rc_post:+.2f} after",
          transform=ax_b.transAxes, ha="left", va="top", fontsize=8, color="0.2", bbox=dict(facecolor="white", edgecolor="none", pad=1.5), zorder=6)
ax_b.legend(frameon=False, fontsize=7.5, loc="lower left", ncol=3, handlelength=1.5, columnspacing=1.2)
ax_b.set_title("(b) The five sectors' levels", loc="left", fontproperties=title_fp, color="0.1", pad=6)
ax_b.set_xlim(YEAR_MIN - 0.5, END + 0.5)
ax_b.set_xticks(range(1980, END + 1, 5))

# a little headroom so the in-panel notes clear the data
for ax in (ax_a, ax_b):
    lo, hi = ax.get_ylim(); ax.set_ylim(lo - 0.12 * (hi - lo), hi + 0.18 * (hi - lo))

fig.tight_layout(h_pad=1.2)
stem = os.path.join(OUTPUT_DIR, "fig_level_vs_amplitude" + ("" if END == 2025 else f"_to{END}"))
ch3_style.savefig_pair(fig, stem, dpi=220)
plt.close(fig)
