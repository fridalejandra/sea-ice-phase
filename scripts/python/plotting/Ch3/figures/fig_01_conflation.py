"""
fig01_concept_manuscript.py — Figure 1, the phase-vs-amplitude conflation concept.
====================================================================================
2x2 manuscript figure illustrating why phase and amplitude must be decomposed
separately: a phase shift (earlier timing, same magnitude) and an amplitude
change (smaller cycle, same timing) can produce a similar summer minimum SIE,
even though the underlying mechanism is completely different.

This is a SCHEMATIC, not a results figure: panels (b)/(c)/(d) apply synthetic
perturbations (PHASE_SHIFT, AMP_SCALE) to the real fitted invariant annual
cycle, purely to illustrate the concept motivating the APAC decomposition.
Sector/period are fixed to one representative curve (circumpolar, full record)
rather than looping — the shape, not the exact numbers, is the point.

Fixed 2026-09 from the previous version, which:
  - referenced a `fitted_invariant` column that no longer exists (the real
    invariant-cycle column is `iac_notrend`; independently confirmed by
    patch_compute_scripts.py's identical fix to compute_phase_amplitude_monthly.py)
  - averaged `iac_notrend` across ALL sectors and (after the dual-period
    consolidation) both periods at once, with no sector/period filter
  - hardcoded the old cluster-only paths and a Drive destination that had
    drifted from ch3_config.py's real GDRIVE constant

Font fix 2026-09-18: this script used to set its own font.family (with
"DejaVu Sans" first in the list -- i.e. it was never actually rendering in
Helvetica even though Helvetica was listed as a fallback) via its own
rcParams.update call. Font family/spines now come from ch3_style (the shared
module every other figure uses), imported below; this script's other, non-
font-family style choices -- the larger font.size/labelsize/titlesize for
this more illustrative/schematic figure, legend and dpi/savefig settings --
are kept as a local rcParams.update placed AFTER the ch3_style import, since
those aren't part of what ch3_style standardizes.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

import ch3_data as D
from ch3_plot import figsize, save
import ch3_style  # sets font (Helvetica/Tacoma) + spines for every figure

# =============================================================================
# STYLE (non-font-family settings only -- font.family/font.sans-serif and
# axes.spines.top/right are ch3_style's job now; see the note above)
# =============================================================================

plt.rcParams.update({
    "font.size"        : 11,
    "axes.linewidth"   : 0.8,
    "axes.labelsize"   : 13,
    "axes.titlesize"   : 13,
    "axes.titleweight" : "normal",
    "xtick.labelsize"  : 11,
    "ytick.labelsize"  : 11,
    "legend.fontsize"  : 10,
    "legend.frameon"   : False,
    "figure.dpi"       : 150,
    "savefig.dpi"      : 300,
    "savefig.bbox"     : "tight",
    "savefig.facecolor": "white",
})

C_INV   = "#2C2C2A"
C_PHASE = "#D4537E"
C_AMP   = "#1D9E75"
C_ANNOT = "#5F5E5A"

SECTOR      = "SIE_circumpolar"   # representative curve for the schematic
PERIOD      = "FULL"
PHASE_SHIFT = 15    # days earlier
AMP_SCALE   = 0.78  # fraction of original range (so ~22% smaller)

MONTH_DAYS   = [0, 28, 59, 89, 120, 150, 181, 212, 242, 273, 303, 334]
MONTH_LABELS = ["Feb","Mar","Apr","May","Jun","Jul",
                "Aug","Sep","Oct","Nov","Dec","Jan"]

# =============================================================================
# BUILD INVARIANT CYCLE CENTRED ON MINIMUM
# =============================================================================

print("Loading data...")
daily = D.load_daily(period=PERIOD)
D.summary(daily=daily)
sec = daily[daily["sector"] == SECTOR]
if sec.empty:
    raise SystemExit(f"No rows for sector={SECTOR!r}, period={PERIOD!r} in daily_fitted.csv")

inv_by_doy = sec.groupby("DOY")["iac_notrend"].mean()
min_doy    = int(inv_by_doy.idxmin())

doys   = np.arange(1, 366)
vals   = np.array([inv_by_doy[d] for d in doys if d in inv_by_doy.index])
if len(vals) != 365:
    # leap-year DOY 366 or a missing DOY; reindex defensively rather than
    # silently misaligning the day axis
    vals = inv_by_doy.reindex(doys).interpolate(limit_direction="both").values
shift  = min_doy - 1

# Centre on minimum and smooth
vals_c = gaussian_filter1d(np.roll(vals, -shift), sigma=2)
N      = len(vals_c)
days   = np.arange(N)

# --- Phase shift: interpolate onto a shifted time axis ---
# Instead of rolling (which wraps), we interpolate the same curve
# evaluated at earlier days — keeping full 365-day coverage
days_shifted = days + PHASE_SHIFT
# Wrap shifted days back into [0, N) for periodic interpolation
days_shifted_wrapped = days_shifted % N
phase_vals = np.interp(days_shifted_wrapped, days, vals_c)
# Re-smooth slightly to remove any interpolation artifacts
phase_vals = gaussian_filter1d(phase_vals, sigma=1)

# --- Amplitude change: scale the range ---
inv_min   = vals_c.min()
inv_range = vals_c.max() - inv_min
amp_vals  = inv_min + (vals_c - inv_min) * AMP_SCALE

def find_right_min(v, start=280):
    return start + int(np.argmin(v[start:]))

# =============================================================================
# 2x2 MANUSCRIPT FIGURE
# =============================================================================

print("Building 2x2 concept figure...")

fig, axes = plt.subplots(2, 2, figsize=figsize("grid2x2"), sharex=True, sharey=True)
axes = axes.flatten()

YMIN = vals_c.min() - 0.1
# Extra headroom vs. the original +0.6 -- panel (b)'s "N days earlier" arrow
# used to sit almost on top of the curve peaks (it was drawn at 97-103% of
# vals_c.max(), which is basically AT the peak); it's now drawn at a fixed
# offset above the peak instead (see panel B below), and needs the room.
YMAX = vals_c.max() + 1.15
panel_labels = ["(a)", "(b)", "(c)", "(d)"]
titles = [
    "Invariant annual cycle",
    "Phase shift — earlier timing, same magnitude",
    "Amplitude change — smaller cycle, same timing",
    "Same summer minimum — different mechanisms",
]

for ax, label, title in zip(axes, panel_labels, titles):
    ax.set_facecolor("white")
    ax.set_xlim(0, N - 1)
    ax.set_ylim(YMIN, YMAX)
    ax.set_xticks(MONTH_DAYS)
    ax.set_xticklabels(MONTH_LABELS, fontsize=10)
    ax.tick_params(labelsize=10)
    # Clean, round, whole-number ticks every 2 million km^2 -- the previous
    # version stepped by 1.0 but still formatted with one decimal place
    # (e.g. "19.0"), which read as more precision than the figure needs and
    # made the axis busy. Step of 2 + integer labels is plainly legible.
    YSTEP = 2.0
    ytick_lo = np.ceil(YMIN / YSTEP) * YSTEP
    yticks = np.arange(ytick_lo, YMAX, YSTEP)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{int(y)}" for y in yticks], fontsize=10)
    ax.text(0.02, 0.97, label, transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top")
    ax.set_title(title, fontsize=10, pad=8, loc="left", color=C_ANNOT)

# Y axis labels on left column only
axes[0].set_ylabel("SIE (million km²)", fontsize=11)
axes[2].set_ylabel("SIE (million km²)", fontsize=11)

# X axis labels on bottom row only
for ax in [axes[2], axes[3]]:
    ax.set_xlabel(
        "Month  (day 0 = annual minimum, late February)",
        fontsize=10, color=C_ANNOT)

# --- Panel A: Invariant cycle ---
ax = axes[0]
ax.plot(days, vals_c, color=C_INV, lw=2.5, zorder=4)
ax.fill_between(days, vals_c, YMIN, color=C_INV, alpha=0.07)

# --- Panel B: Phase shift ---
ax = axes[1]
ax.plot(days, vals_c,     color=C_INV,   lw=2.0, alpha=0.5,
        zorder=4, label="Invariant cycle")
ax.plot(days, phase_vals, color=C_PHASE, lw=2.5, zorder=5,
        linestyle="--", label="Phase shifted (earlier)")

# Mark peaks with vertical dotted lines
peak_inv   = int(np.argmax(vals_c))
peak_phase = int(np.argmax(phase_vals))
ax.axvline(peak_inv,   color=C_INV,   lw=1.0, linestyle=":", alpha=0.5)
ax.axvline(peak_phase, color=C_PHASE, lw=1.0, linestyle=":", alpha=0.7)

# Annotate the shift. Fixed data-unit offsets above the curves' own max,
# not a multiple of vals_c.max() (0.97x/1.03x put the arrow almost exactly
# at the peaks' height, since that's what vals_c.max() IS -- the arrow and
# the "N days earlier" text were overlapping the curve peaks themselves).
ax.annotate("",
    xy=(peak_phase, vals_c.max() + 0.55),
    xytext=(peak_inv, vals_c.max() + 0.55),
    arrowprops=dict(arrowstyle="<->", color=C_ANNOT, lw=1.1)
)
ax.text((peak_phase + peak_inv) / 2, vals_c.max() + 0.8,
        f"~{PHASE_SHIFT} days earlier",
        ha="center", va="bottom", fontsize=8, color=C_ANNOT)

ax.legend(fontsize=9, frameon=False, loc="upper left",
          bbox_to_anchor=(0.02, 0.88))
ax = axes[2]
ax.plot(days, vals_c,   color=C_INV, lw=2.0, alpha=0.5,
        zorder=4, label="Invariant cycle")
ax.plot(days, amp_vals, color=C_AMP, lw=2.5, zorder=5,
        linestyle="--", label="Reduced amplitude")

# Mark shared peak timing
peak_inv = int(np.argmax(vals_c))
ax.axvline(peak_inv, color=C_ANNOT, lw=0.8, linestyle=":", alpha=0.4)

# Annotate amplitude difference
ax.annotate("",
    xy=(peak_inv + 5, amp_vals.max()),
    xytext=(peak_inv + 5, vals_c.max()),
    arrowprops=dict(arrowstyle="<->", color=C_ANNOT, lw=1.1)
)
ax.text(peak_inv + 18, (amp_vals.max() + vals_c.max()) / 2,
        "smaller\namplitude",
        ha="left", va="center", fontsize=8, color=C_ANNOT)

ax.legend(fontsize=9, frameon=False, loc="upper left",
          bbox_to_anchor=(0.02, 0.88))
ax = axes[3]
ax.plot(days, vals_c,     color=C_INV,   lw=2.0, alpha=0.5,
        zorder=4, label="Invariant cycle")
ax.plot(days, phase_vals, color=C_PHASE, lw=2.5, zorder=5,
        linestyle="--", label="Phase shifted")
ax.plot(days, amp_vals,   color=C_AMP,   lw=2.5, zorder=5,
        linestyle="--", label="Reduced amplitude")

# Mark summer minima with dots
for v, color in [(vals_c, C_INV), (phase_vals, C_PHASE), (amp_vals, C_AMP)]:
    d = find_right_min(v)
    ax.scatter(d, v[d], color=color, s=80, zorder=7,
               edgecolors="white", linewidth=1.5)

# Annotate convergence of minima. The three minima all fall late in the
# cycle (find_right_min searches from day 280 onward, close to the Jan
# wrap), so a label centred between them and placed right above them --
# the original approach -- sits almost exactly at the panel's right edge,
# on top of the last x tick label. Anchor the text well inside the panel
# instead (bounded so it can't drift past the edge either) and point a
# single arrow at the cluster of dots.
min_inv   = find_right_min(vals_c)
min_phase = find_right_min(phase_vals)
min_amp   = find_right_min(amp_vals)

cluster_x = (min_phase + min_amp) / 2
cluster_y = max(vals_c[min_inv], phase_vals[min_phase], amp_vals[min_amp])
label_x = min(max(N * 0.60, cluster_x - 55), N - 60)
label_y = cluster_y + 1.1

ax.annotate("Similar minimum SIE",
    xy=(cluster_x, cluster_y + 0.15), xycoords="data",
    xytext=(label_x, label_y), textcoords="data",
    ha="center", va="bottom", fontsize=8, color=C_ANNOT, style="italic",
    arrowprops=dict(arrowstyle="->", color=C_ANNOT, lw=0.9, alpha=0.75,
                     connectionstyle="arc3,rad=0.15"))

ax.legend(fontsize=9, frameon=False, loc="upper left",
          bbox_to_anchor=(0.02, 0.88))

fig.tight_layout()
fig.subplots_adjust(hspace=0.38, wspace=0.18)

save(fig, "fig01_concept_manuscript.png", sync=True)
print("Done.")