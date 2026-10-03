#!/usr/bin/env python3
"""
fig1_cycle_mechanisms.py -- Fig. 1 (Introduction): one panel, one point.
A cycle that runs early, a cycle with a smaller range, and a cycle that sits
lower are three different changes, yet on a given day each departs from the
invariant cycle by the same amount -- one anomaly, three changes.

v5 (2026-10-04). Cut down from v4's four panels (phase / amplitude / level /
"which measure registers which change" matrix). The matrix and the
"minimum held at the floor" amplitude variant were moved out: both lean on
measures defined in Methods and on a hypothesis that belongs to Results, so
neither belongs in the Introduction. v3 (the old 2x2 phase/amplitude/
combined layout) is backed up in the session scratchpad if ever needed.

The three change sizes are not arbitrary: each is SOLVED so that its
anomaly on MARK_DAY equals ANOMALY exactly (see solve_* below). That is the
whole figure -- the three dots on the marked day land on the same value.
MARK_DAY is in the retreat season, so all three anomalies are negative:
matches the intro's wording ("a cycle that runs early looks ... like a
cycle that fell in level; a cycle with a smaller range can look like one
that simply sits lower").

Schematic only: the invariant cycle is a smooth idealized curve, not fit to
observations. No data dependency -- `python fig1_cycle_mechanisms.py`.

Writes (via ch3_style.savefig_pair):
  figures/fig1_cycle_mechanisms.png              -- white bg, ch3_main.tex
  figures/fig1_cycle_mechanisms_presentation.png -- transparent bg, slides
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ch3_style

BLUE, OR, GREEN, GREY = ch3_style.BLUE, ch3_style.ORANGE, ch3_style.GREEN, ch3_style.GREY
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

LO, HI = 0.15, 1.0      # invariant cycle's minimum and maximum (arbitrary units)
MARK_DAY = 258          # a retreat-season day (~Nov on this axis)
ANOMALY = -0.08         # the shared anomaly all three changes produce on MARK_DAY


def invariant(d):
    return LO + (HI - LO) * (1 - np.cos(2 * np.pi * (d - 15) / 365)) / 2


MEAN = (LO + HI) / 2
base = invariant(MARK_DAY)

# solve each change's size so its anomaly on MARK_DAY is exactly ANOMALY
SHIFT = brentq(lambda k: invariant(MARK_DAY + k) - base - ANOMALY, 0.1, 60)   # days earlier
SCALE = 1 + ANOMALY / (base - MEAN)                                          # range fraction
LEVEL = ANOMALY                                                              # vertical offset


def earlier(d):
    return invariant(d + SHIFT)


def smaller(d):
    return MEAN + SCALE * (invariant(d) - MEAN)


def lower(d):
    return invariant(d) + LEVEL


# direct labels placed in open space (label x, label y, alignment), not on the curves
CURVES = [  # function, colour, direct label, (x, y, ha)
    (earlier, BLUE,  f"runs {SHIFT:.0f} days earlier", (112, 0.95, "right")),
    (smaller, OR,    "smaller range", (8, 0.49, "left")),   # size is in the caption
    (lower,   GREEN, "sits lower", (352, -0.035, "right")),
]
for f, _, lab, _ in CURVES:
    print(f"  {lab:28s} anomaly on day {MARK_DAY}: {f(MARK_DAY) - base:+.3f}")
print(f"  (for the caption: {SHIFT:.0f} days earlier; range {1 - SCALE:.0%} smaller; "
      f"level {LEVEL:+.2f} = {-LEVEL / (HI - LO):.0%} of the invariant range lower)")

doy = np.linspace(0, 365, 800)
fig, ax = plt.subplots(figsize=(8.5, 4.8))

ax.plot(doy, invariant(doy), color=GREY, lw=2.0, ls=(0, (5, 3)), zorder=2)
ax.text(205, invariant(205) + 0.035, "invariant cycle", color="0.4", fontsize=9, ha="left")
for f, colour, lab, (xl, yl, ha) in CURVES:
    ax.plot(doy, f(doy), color=colour, lw=2.4, zorder=3)
    ax.text(xl, yl, lab, color=colour, ha=ha, va="center",
            fontproperties=ch3_style.bold_font_properties(size=9.5))

# the marked day: three different changes, one anomaly
ax.axvline(MARK_DAY, color="0.75", lw=1, ls=":", zorder=1)
ax.scatter([MARK_DAY], [base], s=46, color="white", edgecolor=GREY, lw=1.6, zorder=5)
ax.scatter([MARK_DAY], [base + ANOMALY], s=60, color="0.15", zorder=6)
ax.annotate("", xy=(MARK_DAY + 6, base + ANOMALY), xytext=(MARK_DAY + 6, base),
            arrowprops=dict(arrowstyle="->", color="0.15", lw=1.4))
ax.text(MARK_DAY + 11, base + ANOMALY / 2,
        "same anomaly on this day\nfrom all three changes", fontsize=9, color="0.2",
        ha="left", va="center")

months = ["Feb", "Apr", "Jun", "Aug", "Oct", "Dec", "Feb"]
ax.set_xticks(np.linspace(0, 365, 7))
ax.set_xticklabels(months, fontsize=9, color="0.4")
ax.set_xlim(0, 365)
ax.set_yticks([])
ax.set_ylabel("Sea ice extent", color="0.4", fontsize=9.5)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.spines["bottom"].set_color("0.7")
ax.set_ylim(-0.09, 1.12)

fig.tight_layout()
ch3_style.savefig_pair(fig, os.path.join(OUT, "fig1_cycle_mechanisms"))
plt.close(fig)
print("ok - fig1 v5 (single panel)")
