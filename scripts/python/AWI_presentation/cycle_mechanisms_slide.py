#!/usr/bin/env python3
"""
cycle_mechanisms_slide.py -- slide version of Ch. 3 Fig. 1 (fig1_cycle_mechanisms.py).

Same figure, same maths: a cycle that runs early, a cycle with a smaller range
and a cycle that sits lower are three different changes, yet on the marked day
each departs from the fixed cycle by the same amount. Each change is SOLVED so
its anomaly on MARK_DAY equals ANOMALY, so the three dots land on one value.

Changes for the talk:
  - standalone: no ch3_style / ch3_config needed (colours and fonts set below)
  - slide-sized text and line weights
  - transparent background, no title; PowerPoint supplies those
  - no numbers on the labels (set SHOW_NUMBERS = True to bring them back);
    the schematic is not a result, so "12 days" should not read like one
  - colours moved away from the blue / orange-red used for the anomaly plot
    on slide 1, so the two figures do not share meanings
  - key at the bottom instead of labels on the curves; labels use the
    chapter's terms (ahead of phase, lower amplitude, lower level)
  - --build writes the two-curve version (no level) for a Keynote reveal

Usage:
    python cycle_mechanisms_slide.py
    python cycle_mechanisms_slide.py --out cycle_mechanisms --width 11 --height 5.6
    python cycle_mechanisms_slide.py --build --out cycle_mechanisms_build1

Writes {out}.svg and {out}.png (300 dpi, transparent).
"""
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# ---- look -----------------------------------------------------------------
INK = "#222222"          # text and axes; use "#EEEEEE" on a dark slide
MUTED = "#6B6B6B"        # secondary text, climatology line
EARLIER = "#CC79A7"      # reddish purple (Okabe-Ito)
SMALLER = "#E69F00"      # amber (Okabe-Ito)
LOWER = "#009E73"        # bluish green (Okabe-Ito)
FONT = None              # e.g. "Arial"; None keeps the matplotlib default
SHOW_NUMBERS = False     # True -> "runs 12 days earlier"
CLIM_LABEL = "fixed climatology"   # Ch. 3 calls this "invariant cycle"

# ---- schematic (same as the chapter figure) -------------------------------
LO, HI = 0.15, 1.0       # fixed cycle's minimum and maximum, arbitrary units
MARK_DAY = 258           # a retreat-season day, so all anomalies are negative
ANOMALY = -0.08          # the shared anomaly all three changes produce


def climatology(d):
    return LO + (HI - LO) * (1 - np.cos(2 * np.pi * (d - 15) / 365)) / 2


MEAN = (LO + HI) / 2
BASE = climatology(MARK_DAY)

SHIFT = brentq(lambda k: climatology(MARK_DAY + k) - BASE - ANOMALY, 0.1, 60)  # days earlier
SCALE = 1 + ANOMALY / (BASE - MEAN)                                          # range fraction
LEVEL = ANOMALY                                                              # vertical offset


def earlier(d):
    return climatology(d + SHIFT)


def smaller(d):
    return MEAN + SCALE * (climatology(d) - MEAN)


def lower(d):
    return climatology(d) + LEVEL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="cycle_mechanisms")
    ap.add_argument("--width", type=float, default=11.0)
    ap.add_argument("--height", type=float, default=5.6)
    ap.add_argument("--build", action="store_true",
                    help="omit the level curve (first step of a Keynote build)")
    args = ap.parse_args()

    if FONT:
        matplotlib.rcParams["font.family"] = FONT
    matplotlib.rcParams.update({
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK,
    })

    # proper terms first (as in Ch. 3), plain gloss second
    earlier_label = (f"ahead of phase ({SHIFT:.0f} days)" if SHOW_NUMBERS
                     else "ahead of phase")
    smaller_label = (f"lower amplitude ({1 - SCALE:.0%})" if SHOW_NUMBERS
                     else "lower amplitude")
    lower_label = "lower level"

    # key at the bottom instead of direct labels (easier to read on a slide)
    curves = [
        (earlier, EARLIER, earlier_label),
        (smaller, SMALLER, smaller_label),
        (lower, LOWER, lower_label),
    ]
    if args.build:                       # Keynote build: level curve appears on the next click
        curves = curves[:2]

    doy = np.linspace(0, 365, 800)
    fig, ax = plt.subplots(figsize=(args.width, args.height))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    ax.plot(doy, climatology(doy), color=MUTED, lw=3.0, ls=(0, (5, 3)), zorder=2)
    ax.text(208, climatology(208) + 0.045, CLIM_LABEL, color=MUTED, fontsize=15,
            ha="left", va="bottom")

    for f, colour, lab in curves:
        ax.plot(doy, f(doy), color=colour, lw=4.0, zorder=3, solid_capstyle="round")

    # the marked day: three different changes, one anomaly
    ax.axvline(MARK_DAY, color="0.7", lw=1.4, ls=":", zorder=1)
    ax.scatter([MARK_DAY], [BASE], s=90, color="white", edgecolor=MUTED, lw=2.2, zorder=5)
    ax.scatter([MARK_DAY], [BASE + ANOMALY], s=110, color=INK, zorder=6)
    ax.annotate("", xy=(MARK_DAY + 7, BASE + ANOMALY), xytext=(MARK_DAY + 7, BASE),
                arrowprops=dict(arrowstyle="->", color=INK, lw=2.0))
    ax.text(MARK_DAY + 14, BASE + ANOMALY / 2, "same anomaly", fontsize=15, color=INK,
            ha="left", va="center")

    ax.set_xticks(np.linspace(0, 365, 7))
    ax.set_xticklabels(["Feb", "Apr", "Jun", "Aug", "Oct", "Dec", "Feb"], fontsize=15)
    ax.set_xlim(0, 365)
    ax.set_ylim(-0.10, 1.18)
    ax.set_yticks([])
    ax.set_ylabel("Sea ice extent", fontsize=16, color=MUTED)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("0.6")
    ax.tick_params(axis="x", length=4)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=c, lw=5, label=lab) for _, c, lab in curves]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.1),
              ncol=3, frameon=False, fontsize=17, handlelength=2.2, columnspacing=2.5)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.96, bottom=0.2)

    fig.savefig(f"{args.out}.svg", transparent=True)
    fig.savefig(f"{args.out}.png", dpi=300, transparent=True)

    for f, _, lab in curves:
        print(f"  {lab:24s} anomaly on day {MARK_DAY}: {f(MARK_DAY) - BASE:+.3f}")
    print(f"wrote {args.out}.svg/.png  (shift {SHIFT:.0f} d, range {1 - SCALE:.0%} smaller, "
          f"level {LEVEL:+.2f})")


if __name__ == "__main__":
    main()