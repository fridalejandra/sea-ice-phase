#!/usr/bin/env python3
"""
fig_phase_adjustment.py -- Methods figure: the Beta-CDF timing warp used by
the phase adjustment, and its effect on the fitted annual cycle.

Schematic only -- Beta(1, 1.6) / Beta(1.6, 1) are illustrative parameters,
not fitted to any particular year. No external data dependency, so this is
fully reproducible from this script alone: run `python fig_phase_adjustment.py`.

Colours come from ch3_style (house palette + font rules); see that module's
docstring for why (Frida's colorblind-safe / cblind request, 2026-09-30).
Writes two files via ch3_style.savefig_pair:
  fig_phase_adjustment.png              -- white bg, wired into ch3_main.tex
  fig_phase_adjustment_presentation.png -- transparent bg, for slides
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ch3_style

BLUE, OR, GREY = ch3_style.BLUE, ch3_style.OR, ch3_style.GREY
from ch3_config import OUTPUT_DIR as OUT  # <repo>/results/ch3/figures, same folder as every other Ch3 figure (was HERE/figures, i.e. inside scripts/ -- fixed 2026-10-04)
os.makedirs(OUT, exist_ok=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

# ---- Panel A: the Beta CDF warp itself ----
t = np.linspace(0, 1, 400)
identity = t
warp_late  = beta_dist.cdf(t, 1.6, 1.0)   # stretches the early part -> effectively delays middle/late landmarks
warp_early = beta_dist.cdf(t, 1.0, 1.6)   # compresses the early part -> effectively advances middle/late landmarks

ax1.plot(t, identity, color=GREY, lw=1.8, ls=(0,(5,3)), label=r"no warp ($\alpha=\beta=1$)")
ax1.plot(t, warp_early, color=BLUE, lw=2.4, label=r"earlier warp ($\alpha=1,\ \beta=1.6$)")
ax1.plot(t, warp_late, color=OR, lw=2.4, label=r"later warp ($\alpha=1.6,\ \beta=1$)")
ax1.scatter([0,1],[0,1], color="0.15", zorder=5, s=30)
ax1.annotate("both ends pinned\n(min to min)", xy=(1,1), xytext=(0.62,0.82),
             fontsize=8.8, color="0.25", arrowprops=dict(arrowstyle="->", color="0.5"))
ax1.set_xlabel("fraction of the cycle elapsed\n(observed, min to min)", fontsize=9.5, color="0.35")
ax1.set_ylabel("warped fraction\n(input to the fitted shape)", fontsize=9.5, color="0.35")
ax1.set_xlim(-0.02,1.02); ax1.set_ylim(-0.02,1.05)
ax1.spines[['top','right']].set_visible(False)
ax1.legend(fontsize=8.3, frameon=False, loc="upper left")
ax1.set_title("(a) The Beta-CDF warp", loc="left", fontsize=12, fontweight="bold", color="0.15", pad=9)

# ---- Panel B: what this does to the actual annual cycle ----
doy = np.linspace(0, 365, 800)
def invariant(d):
    return (1 - np.cos(2*np.pi*(d-15)/365)) / 2

# map calendar day -> fraction elapsed since the anchor minimum (day 15), 0..1 over the 365-day cycle
frac = ((doy - 15) % 365) / 365
warped_frac_early = beta_dist.cdf(frac, 1.0, 1.6)
warped_frac_late  = beta_dist.cdf(frac, 1.6, 1.0)
# convert warped fraction back to an "effective day of year" for plotting against the common shape
eff_day_early = 15 + warped_frac_early * 365
eff_day_late  = 15 + warped_frac_late * 365

shape_on_warped_early = invariant(eff_day_early)
shape_on_warped_late  = invariant(eff_day_late)

ax2.plot(doy, invariant(doy), color=GREY, lw=1.8, ls=(0,(5,3)), label="common shape (no warp)")
ax2.plot(doy, shape_on_warped_early, color=BLUE, lw=2.4, label="earlier-warped year")
ax2.plot(doy, shape_on_warped_late, color=OR, lw=2.4, label="later-warped year")
ax2.scatter([15,15],[invariant(15),invariant(15)], color="0.15", zorder=5, s=34)
ax2.annotate("minimum date fixed\nby construction", xy=(15, invariant(15)), xytext=(70,0.15),
             fontsize=8.8, color="0.25", arrowprops=dict(arrowstyle="->", color="0.5"))
maxday_inv = doy[np.argmax(invariant(doy))]
maxday_e = doy[np.argmax(shape_on_warped_early)]
maxday_l = doy[np.argmax(shape_on_warped_late)]
ax2.annotate("", xy=(maxday_e, 1.03), xytext=(maxday_inv, 1.03),
             arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.4))
ax2.annotate("", xy=(maxday_l, 1.08), xytext=(maxday_inv, 1.08),
             arrowprops=dict(arrowstyle="->", color=OR, lw=1.4))
ax2.text((maxday_inv+maxday_e)/2, 1.045, "max shifts earlier", fontsize=8, color=BLUE, ha="center")
ax2.text((maxday_inv+maxday_l)/2, 1.10, "max shifts later", fontsize=8, color=OR, ha="center")
months = ["Feb","Apr","Jun","Aug","Oct","Dec","Feb"]
ax2.set_xticks(np.linspace(0,365,7)); ax2.set_xticklabels(months, fontsize=9, color="0.4")
ax2.set_yticks([])
ax2.set_ylim(-0.05, 1.18)
ax2.spines[['top','right','left']].set_visible(False)
ax2.spines['bottom'].set_color("0.7")
ax2.legend(fontsize=8.3, frameon=False, loc="upper left")
ax2.set_title("(b) Effect on the fitted cycle", loc="left", fontsize=12, fontweight="bold", color="0.15", pad=9)

fig.suptitle("The phase adjustment warps timing while leaving the anchor minimum fixed",
             fontsize=13, x=0.01, ha="left", y=1.02)
fig.text(0.01, -0.05,
    "(a) Each year's cycle is measured as the fraction of time elapsed since its observed minimum, then passed through a\n"
    "two-parameter Beta CDF that stretches some parts of the cycle and compresses others, leaving both endpoints (0 and 1) fixed.\n"
    "(b) Applied to the common shape, the warp moves the timing of everything between the two minima, most visibly the\n"
    "maximum, while the minimum date itself cannot move, by construction. Schematic parameters, illustrative only.",
    fontsize=8.6, color="0.4")

fig.tight_layout(rect=[0,0.05,1,0.93])
ch3_style.savefig_pair(fig, os.path.join(OUT, "fig_phase_adjustment"))
print("ok - phase adjustment diagram (ch3_style palette, dual export)")
