#!/usr/bin/env python
import os, sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import seaborn as sns
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ch4_figstyle as fs
OUT = "/user/geog/falejandraperez/sea-ice-phase/results/ch3/figures/manuscript"
os.makedirs(OUT, exist_ok=True)
cb = sns.color_palette("colorblind")
C_BLUE, C_ORANGE, C_GREEN, C_GREY = cb[0], cb[1], cb[2], "#888"

t = np.arange(1, 366)
peak_doy, mean_lvl, amp = 227, 10.0, 9.0
# shift=+12 makes blue peak at t=215, i.e., 12 days EARLIER than invariant.
# scale=0.688 and offset=-1.60 make all three alternative cycles cross
# the invariant with the same (-1.60) departure on day 283.
def cyc(shift=0, scale=1.0, offset=0.0):
    return mean_lvl + offset + scale*amp*np.cos(2*np.pi*(t - peak_doy + shift)/365)
invariant = cyc(); blue = cyc(shift=12); gold = cyc(scale=0.688); green = cyc(offset=-1.60)

fs.set_mode("manuscript")
fig, ax = plt.subplots(figsize=fs.size(1.0, 0.60))

# subtle horizontal reference lines at invariant max and min
inv_max, inv_min = invariant.max(), invariant.min()
ax.axhline(inv_max, ls=(0, (1.5, 3)), color=C_GREY, lw=0.6, alpha=0.5)
ax.axhline(inv_min, ls=(0, (1.5, 3)), color=C_GREY, lw=0.6, alpha=0.5)

# four cycles
ax.plot(t, invariant, "--", color=C_GREY, lw=1.4)
ax.plot(t, blue,  color=C_BLUE,   lw=1.7)
ax.plot(t, gold,  color=C_ORANGE, lw=1.7)
ax.plot(t, green, color=C_GREEN,  lw=1.7)

# inline labels placed to avoid overlap
ax.text(140, 14.0, "runs 12 days earlier", color=C_BLUE,   fontsize=11, ha="center")
ax.text(225, inv_max + 1.1, "invariant cycle", color=C_GREY,   fontsize=10.5, ha="center")
ax.text(55,  5.2,          "smaller range",      color=C_ORANGE, fontsize=10.5, ha="left")
ax.text(55,  -1.6,         "sits lower",         color=C_GREEN,  fontsize=10.5, ha="left")

# marked day in retreat season -- open ring on invariant, filled dot at common point
mday = 283
ax.axvline(mday, ls=":", color="#444", lw=0.6)
ax.plot(mday, invariant[mday-1], "o", markerfacecolor="none",
        markeredgecolor="black", ms=6, mew=1.0)
ax.plot(mday, gold[mday-1],      "o", color="black", ms=5, mec="white", mew=0.6)
ax.annotate("same anomaly on this day\nfrom three different cycles",
            xy=(mday+2, gold[mday-1]),
            xytext=(mday+10, gold[mday-1]-2.0),
            fontsize=9.5, ha="left", va="center", color="#333",
            arrowprops=dict(arrowstyle="-", color="#777", lw=0.5))

# axes
month_doy = [1, 60, 121, 182, 244, 305, 365]
ax.set_xticks(month_doy); ax.set_xticklabels(["Feb","Apr","Jun","Aug","Oct","Dec","Feb"])
ax.set_xlim(1, 365); ax.set_ylim(-3, 23)
ax.set_ylabel("Sea ice extent"); ax.set_yticks([])
fs.FIG_ROOT = os.path.dirname(OUT); fs.save(fig, "fig1_cycle_mechanisms")
print(f"wrote {OUT}/fig1_cycle_mechanisms.png")
