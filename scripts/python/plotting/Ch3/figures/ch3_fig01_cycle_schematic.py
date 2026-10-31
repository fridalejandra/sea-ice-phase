#!/usr/bin/env python
"""ch3_fig01_cycle_schematic.py -- Fig. 3.1: three changes to the cycle, one anomaly.
Schematic (no data). Blue runs 12 days earlier, orange has a range 31% smaller,
green sits lower; all three cross the invariant cycle with the same departure on
day 283. Single panel, so no letter."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ch3_style as S

t = np.arange(1, 366)
PEAK, MEAN, AMP, MDAY = 227, 10.0, 9.0, 283


def cyc(shift=0, scale=1.0, offset=0.0):
    return MEAN + offset + scale * AMP * np.cos(2 * np.pi * (t - PEAK + shift) / 365)


inv, early, small, low = cyc(), cyc(shift=12), cyc(scale=0.688), cyc(offset=-1.60)

fig, ax = plt.subplots(figsize=(S.TEXT_WIDTH_IN, 3.6))
for y in (inv.max(), inv.min()):
    ax.axhline(y, ls=(0, (1.5, 3)), color=S.GREY, lw=0.6, alpha=0.6)
ax.plot(t, inv, "--", color=S.GREY, lw=1.4, label="invariant cycle")
ax.plot(t, early, color=S.BLUE, label="runs 12 days earlier (phase)")
ax.plot(t, small, color=S.ORANGE, label="smaller range (amplitude)")
ax.plot(t, low, color=S.GREEN, label="sits lower (annual mean)")
ax.legend(loc="upper left", fontsize=7.5, handlelength=2.2)

ax.axvline(MDAY, ls=":", color=S.INK, lw=0.6)
ax.plot(MDAY, inv[MDAY - 1], "o", mfc="none", mec=S.INK, ms=6, mew=1.0)
ax.plot(MDAY, small[MDAY - 1], "o", color=S.INK, ms=5, mec="white", mew=0.6)
ax.annotate("same anomaly on this day\nfrom three different cycles",
            xy=(MDAY + 2, small[MDAY - 1]), xytext=(362, 21.0),
            fontsize=7.5, ha="right", va="center", color=S.INK,
            arrowprops=dict(arrowstyle="-", color=S.GREY, lw=0.6))

ax.set_xticks([1, 60, 121, 182, 244, 305, 365])
ax.set_xticklabels(["Feb", "Apr", "Jun", "Aug", "Oct", "Dec", "Feb"])
ax.set_xlim(1, 365); ax.set_ylim(-3, 23)
ax.set_ylabel("Sea ice extent"); ax.set_yticks([])
S.save_figure(fig, "ch3_fig01_cycle_schematic")
