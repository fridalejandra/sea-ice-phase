#!/usr/bin/env python3
"""
level_leak.py -- Methods figure (Combining the Components): why a level
shift confined to one or two years is credited to the amplitude component
rather than the trend term, and why that isn't visible in observed
amplitude. Written as fig_level_leak.png in ch3_main.tex.

Panels A and B are schematic/synthetic (illustrative trend + a synthetic
El Nino-year jump) -- they are not computed from the real annual fit. Panel
C's two correlations (amplitude component vs. Nino3.4 = +0.32; observed
amplitude vs. Nino3.4 = +0.01) ARE the real reported values (Weddell Sea,
1979-2025, detrended) and are currently hardcoded here as `vals = [0.32,
0.01]` rather than read from a CSV -- see PROVENANCE.md in this directory:
check_weddell_enso_components.py computes this same correlation but could
not be re-run in the Claude sandbox this script was retrofitted in (it
reads DAILY_CSV via ch3_config, which resolves to a repo path that only
exists on Frida's own machine, not in the sandbox). Re-run
check_weddell_enso_components.py on your machine to confirm r=+0.32/+0.01
are still current before the next time this figure is regenerated -- the
two numbers below should be updated together with that confirmation, not
independently.

Colours come from ch3_style (house palette + font rules); see that module's
docstring for why (Frida's colorblind-safe / cblind request, 2026-09-30).
Writes two files via ch3_style.savefig_pair:
  fig_level_leak.png              -- white bg, wired into ch3_main.tex
  fig_level_leak_presentation.png -- transparent bg, for slides
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ch3_style

BLUE, OR, GREY, GREY2 = ch3_style.BLUE, ch3_style.OR, ch3_style.GREY, ch3_style.GREY2
from ch3_config import OUTPUT_DIR as OUT  # <repo>/results/ch3/figures, same folder as every other Ch3 figure (was HERE/figures, i.e. inside scripts/ -- fixed 2026-10-04)
os.makedirs(OUT, exist_ok=True)

u = np.linspace(0, 1, 400)

def cyc(level=0.0, amp=1.0):
    return level + amp * (1 - np.cos(2*np.pi*u)) / 2

fig = plt.figure(figsize=(13.2, 4.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.15, 0.95], wspace=0.32)

# ---------------- Panel A: what a level shift looks like ----------------
a = fig.add_subplot(gs[0])
base = cyc()
elnino = cyc(level=0.34)
a.plot(u, base, color=GREY, lw=2.0, ls=(0, (5,3)), label="an ordinary year")
a.plot(u, elnino, color=BLUE, lw=2.6, label="an El Niño year")
a.add_patch(FancyArrowPatch((0.5, base.max()), (0.5, elnino.max()), arrowstyle="-|>",
            mutation_scale=13, color="0.2", lw=1.4, shrinkA=0, shrinkB=0))
a.text(0.53, (base.max()+elnino.max())/2, "max\n+0.30", fontsize=9.3, color="0.2", va="center")
a.add_patch(FancyArrowPatch((0.015, base[0]), (0.015, elnino[0]), arrowstyle="-|>",
            mutation_scale=13, color="0.2", lw=1.4, shrinkA=0, shrinkB=0))
a.text(0.045, (base[0]+elnino[0])/2, "min\n+0.40", fontsize=9.3, color="0.2", va="center")
a.set_xlim(-0.03, 1.05); a.set_ylim(-0.1, 1.65)
a.set_xticks([0, 0.5, 1]); a.set_xticklabels(["Feb", "Aug", "Feb"], fontsize=9, color="0.4")
a.set_yticks([]); a.spines[['top','right','left']].set_visible(False)
a.spines['bottom'].set_color("0.7")
a.legend(fontsize=9, frameon=False, loc="upper right", bbox_to_anchor=(1.02, 1.0))
a.set_title("A level shift moves both ends\nthe same way", loc="left", fontsize=12,
            fontweight="bold", color="0.15", pad=9)

# ---------------- Panel B: why it lands in the amplitude component -------
b = fig.add_subplot(gs[1])
years = np.arange(1988, 2011)
rng = np.random.default_rng(3)
trend = 0.15*np.sin((years-1988)/22*np.pi) + 0.02*(years-1999)/11
level_true = trend + rng.normal(0, 0.05, len(years))
elnino_yr = 2003
idx = np.where(years == elnino_yr)[0][0]
level_true[idx] = trend[idx] + 0.34   # the El Nino jump

b.plot(years, trend, color=GREY, lw=2.2, label="the trend term\n(smooth across years)")
b.scatter(years, level_true, s=34, color="0.55", zorder=3)
b.scatter([years[idx]], [level_true[idx]], s=100, color=BLUE, zorder=4, label="the El Niño year")
b.add_patch(FancyArrowPatch((years[idx]+0.3, trend[idx]), (years[idx]+0.3, level_true[idx]),
            arrowstyle="-|>", mutation_scale=13, color=OR, lw=1.6, shrinkA=0, shrinkB=2))
b.annotate("the trend can't bend for one year\nthis gap is credited to the\namplitude component instead",
           xy=(years[idx]+0.3, (trend[idx]+level_true[idx])/2), xytext=(1996.5, -0.10),
           fontsize=9.3, color="0.2", ha="center", arrowprops=dict(arrowstyle="->", color="0.5"))
b.set_xlabel("year", fontsize=9.5, color="0.4")
b.set_yticks([]); b.spines[['top','right','left']].set_visible(False)
b.spines['bottom'].set_color("0.7")
b.legend(fontsize=8.8, frameon=False, loc="upper right")
b.set_title("The trend term is smooth;\na single year's jump can't fit in it",
            loc="left", fontsize=12, fontweight="bold", color="0.15", pad=9)
b.set_ylim(-0.30, 0.75)

# ---------------- Panel C: two numbers, one event -------------------------
# See module docstring: these are the real reported Weddell/Nino3.4 values,
# not computed in this script. Confirm against check_weddell_enso_components.py
# on a machine where DAILY_CSV actually resolves before trusting a change here.
c = fig.add_subplot(gs[2])
labels = ["amplitude\ncomponent", "observed\namplitude\n(max − min)"]
vals = [0.32, 0.01]
cols = [BLUE, GREY2]
bars = c.bar(range(2), vals, color=cols, width=0.55)
for i, v in enumerate(vals):
    c.text(i, v + 0.02, f"r = {v:+.2f}", ha="center", fontsize=11,
           fontweight="bold" if abs(v) > 0.29 else "normal",
           color="0.15" if abs(v) > 0.29 else "0.45")
c.axhline(0.29, color="0.78", lw=1, ls=(0, (4,3)))
c.text(1.55, 0.29, " p < 0.05", fontsize=8.3, color="0.45", va="center")
c.axhline(0, color="0.3", lw=1)
c.set_xticks(range(2)); c.set_xticklabels(labels, fontsize=9.8)
c.set_ylim(-0.08, 0.5); c.spines[['top','right']].set_visible(False)
c.set_ylabel("correlation with Niño3.4", fontsize=9.5)
c.set_title("Same El Niño years,\ntwo different amplitude numbers",
            loc="left", fontsize=12, fontweight="bold", color="0.15", pad=9)

fig.suptitle("Why a level shift shows up in the amplitude component but not in observed amplitude — the Weddell / Niño3.4 case",
             fontsize=13.5, x=0.01, ha="left", y=1.04)
fig.text(0.01, -0.10,
    "Schematic cycles in panel A are illustrative; panel B uses a synthetic trend to show the mechanism. The correlations in panel C are the reported values\n"
    "(Weddell Sea, annual Niño3.4, 1979–2025, detrended). Because the amplitude-adjusted fit is anchored to each year's own minimum and maximum, and the\n"
    "trend term only follows slow, multi-year change, a level shift confined to one or two years is left for the amplitude component to absorb — even though\n"
    "the observed amplitude (max − min), which cancels a shift affecting both ends equally, shows almost nothing.",
    fontsize=8.6, color="0.4")

fig.tight_layout(rect=[0, 0.02, 1, 0.90])
ch3_style.savefig_pair(fig, os.path.join(OUT, "fig_level_leak"))
print("ok - level leak diagram (ch3_style palette, dual export)")
