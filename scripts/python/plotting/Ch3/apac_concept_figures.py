"""
apac_concept_figures.py
Generates 4 PNG figures illustrating APAC phase vs amplitude decomposition.
Clean white style, Nimbus Sans font.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy.ndimage import gaussian_filter1d
import os

# =============================================================================
# 0. SETTINGS
# =============================================================================

DAILY_CSV  = "/user/geog/falejandraperez/sea-ice-phase/scripts/R/chapter3/data/daily_fitted.csv"
OUTPUT_DIR = "/user/geog/falejandraperez/sea-ice-phase/scripts/R/chapter3/figures/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family":       "Nimbus Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    1.0,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.major.size":  4,
    "ytick.major.size":  4,
    "xtick.direction":   "out",
    "ytick.direction":   "out",
    "figure.dpi":        150,
    "savefig.dpi":       200,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "white",
})

C_INV    = "#2C2C2A"
C_PHASE  = "#D4537E"
C_AMP    = "#1D9E75"
C_THRESH = "#BA7517"
C_ANNOT  = "#5F5E5A"

PHASE_SHIFT = 12
AMP_CHANGE  = -0.18

# =============================================================================
# 1. BUILD INVARIANT CYCLE
# =============================================================================

daily      = pd.read_csv(DAILY_CSV)
inv_by_doy = daily.groupby("DOY")["fitted_invariant"].mean()
min_doy    = int(inv_by_doy.idxmin())

doys  = np.arange(1, 366)
vals  = np.array([inv_by_doy[d] for d in doys])
shift = min_doy - 1
vals_c = gaussian_filter1d(np.roll(vals, -shift), sigma=2)
days   = np.arange(365)

# Month tick positions and labels (days from minimum, ~Feb 18)
MONTH_DAYS   = [0,  28,  59,  89, 120, 150, 181, 212, 242, 273, 303, 334]
MONTH_LABELS = ["Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec","Jan"]

# Synthetic curves
phase_vals = np.roll(vals_c, -PHASE_SHIFT)
inv_min    = vals_c.min()
inv_range  = vals_c.max() - inv_min
amp_vals   = inv_min + (vals_c - inv_min) * ((inv_range + AMP_CHANGE) / inv_range)
thresh_val = inv_min + inv_range * 0.52

# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================

def style_ax(ax, title, ymax=4.3):
    ax.set_facecolor("white")
    ax.set_xlim(-5, 369)
    ax.set_ylim(0.3, ymax)
    ax.set_xticks(MONTH_DAYS)
    ax.set_xticklabels(MONTH_LABELS, fontsize=12)
    ax.set_ylabel("SIE (million km²)", fontsize=12, labelpad=8)
    ax.set_xlabel("Month  (day 0 = annual minimum, late February)",
                  fontsize=10, color=C_ANNOT, labelpad=8)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12, loc="left")
    ax.tick_params(labelsize=11)
    # Clean up y ticks to avoid overlap
    yticks = np.arange(0.5, ymax, 0.5)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{y:.1f}" for y in yticks], fontsize=11)


def stroke(fg="white"):
    return [pe.withStroke(linewidth=3, foreground=fg)]


def find_descent_crossing(vals, thresh, start=150):
    for i in range(start, len(vals) - 1):
        if vals[i] >= thresh >= vals[i + 1]:
            return i
    return None


# =============================================================================
# FIG 1 — Invariant cycle only
# =============================================================================

fig, ax = plt.subplots(figsize=(11, 6))
style_ax(ax, "The invariant annual cycle")

ax.plot(days, vals_c, color=C_INV, lw=2.5, zorder=4)
ax.fill_between(days, vals_c, 0.3, color=C_INV, alpha=0.07)

# Annotate minimum
ax.annotate("Annual minimum\n(day 0 ≈ Feb 18)",
            xy=(2, vals_c[2]),
            xytext=(45, vals_c[2] + 0.6),
            fontsize=10, color=C_ANNOT,
            arrowprops=dict(arrowstyle="->", color=C_ANNOT, lw=0.9),
            path_effects=stroke())

# Annotate maximum
peak_day = int(np.argmax(vals_c))
ax.annotate("Annual maximum\n≈ mid-September",
            xy=(peak_day, vals_c[peak_day]),
            xytext=(peak_day - 60, vals_c[peak_day] + 0.18),
            fontsize=10, color=C_ANNOT,
            arrowprops=dict(arrowstyle="->", color=C_ANNOT, lw=0.9),
            path_effects=stroke())

fig.savefig(os.path.join(OUTPUT_DIR, "apac_concept_fig1_invariant.png"))
plt.close(fig)
print("Fig 1 saved")

# =============================================================================
# FIG 2 — Phase shift
# =============================================================================

fig, ax = plt.subplots(figsize=(11, 6))
style_ax(ax, "Phase shift — the whole cycle moves earlier")

ax.plot(days, vals_c,     color=C_INV,   lw=2.5, alpha=0.6, zorder=4)
ax.plot(days, phase_vals, color=C_PHASE, lw=2.5, zorder=5, linestyle="--")

# Peak shift annotation
peak_inv   = int(np.argmax(vals_c))
peak_phase = int(np.argmax(phase_vals))
ymax_v     = vals_c[peak_inv]

ax.annotate("",
            xy=(peak_phase, ymax_v + 0.28),
            xytext=(peak_inv, ymax_v + 0.28),
            arrowprops=dict(arrowstyle="<->", color=C_ANNOT, lw=1.1))
ax.text((peak_inv + peak_phase) / 2, ymax_v + 0.38,
        f"−{PHASE_SHIFT} days earlier",
        ha="center", fontsize=10, color=C_ANNOT,
        fontweight="bold", path_effects=stroke())

# Same peak height annotation
ax.text(peak_inv + 15, ymax_v + 0.05,
        "Same peak height",
        ha="left", fontsize=10, color=C_ANNOT, path_effects=stroke())

# Curve labels
ax.text(185, vals_c[185] + 0.14,
        "Climatological cycle", color=C_INV,
        fontsize=11, fontweight="bold", path_effects=stroke())
ax.text(163, phase_vals[163] - 0.22,
        "Phase shifted", color=C_PHASE,
        fontsize=11, fontweight="bold", path_effects=stroke())

fig.savefig(os.path.join(OUTPUT_DIR, "apac_concept_fig2_phase.png"))
plt.close(fig)
print("Fig 2 saved")

# =============================================================================
# FIG 3 — Amplitude change
# =============================================================================

fig, ax = plt.subplots(figsize=(11, 6))
style_ax(ax, "Amplitude change — smaller cycle, same timing")

ax.plot(days, vals_c,   color=C_INV,  lw=2.5, alpha=0.6, zorder=4)
ax.plot(days, amp_vals, color=C_AMP,  lw=2.5, zorder=5, linestyle="--")

# Vertical line at peak
ax.axvline(peak_inv, color=C_ANNOT, lw=0.8, linestyle=":", alpha=0.5)

# Amplitude bracket
peak_inv_v = vals_c[peak_inv]
peak_amp_v = amp_vals[peak_inv]
ax.annotate("",
            xy=(peak_inv + 10, peak_amp_v),
            xytext=(peak_inv + 10, peak_inv_v),
            arrowprops=dict(arrowstyle="<->", color=C_ANNOT, lw=1.1))
ax.text(peak_inv + 14, (peak_inv_v + peak_amp_v) / 2,
        f"{AMP_CHANGE:+.2f} Mkm²",
        ha="left", va="center", fontsize=10,
        color=C_ANNOT, fontweight="bold", path_effects=stroke())

ax.text(peak_inv - 45, peak_inv_v + 0.1,
        "Same timing",
        ha="center", fontsize=10, color=C_ANNOT, path_effects=stroke())

# Curve labels
ax.text(185, vals_c[185] + 0.14,
        "Climatological cycle", color=C_INV,
        fontsize=11, fontweight="bold", path_effects=stroke())
ax.text(185, amp_vals[185] - 0.22,
        "Amplitude reduced", color=C_AMP,
        fontsize=11, fontweight="bold", path_effects=stroke())

fig.savefig(os.path.join(OUTPUT_DIR, "apac_concept_fig3_amplitude.png"))
plt.close(fig)
print("Fig 3 saved")

# =============================================================================
# FIG 4 — All three + threshold
# =============================================================================

fig, ax = plt.subplots(figsize=(11, 6))
style_ax(ax, "Both cross the threshold earlier — for different reasons")

ax.plot(days, vals_c,     color=C_INV,   lw=2.5, alpha=0.6, zorder=4)
ax.plot(days, phase_vals, color=C_PHASE, lw=2.5, zorder=5, linestyle="--")
ax.plot(days, amp_vals,   color=C_AMP,   lw=2.5, zorder=5, linestyle="--")

# Threshold
ax.axhline(thresh_val, color=C_THRESH, lw=1.5,
           linestyle=(0, (4, 3)), zorder=3)
ax.text(362, thresh_val + 0.07, "Threshold",
        color=C_THRESH, fontsize=10, fontweight="bold",
        ha="right", path_effects=stroke())

# Crossing dots
cross_inv   = find_descent_crossing(vals_c,     thresh_val)
cross_phase = find_descent_crossing(phase_vals, thresh_val)
cross_amp   = find_descent_crossing(amp_vals,   thresh_val)

for cross, color in [(cross_inv, C_INV),
                     (cross_phase, C_PHASE),
                     (cross_amp, C_AMP)]:
    if cross:
        ax.scatter(cross, thresh_val, color=color, s=70, zorder=7,
                   edgecolors="white", linewidth=1.5)

# Bracket showing both cross earlier
if cross_phase and cross_amp and cross_inv:
    earlier  = min(cross_phase, cross_amp)
    y_brack  = thresh_val - 0.32
    ax.annotate("",
                xy=(earlier, y_brack),
                xytext=(cross_inv, y_brack),
                arrowprops=dict(arrowstyle="<->", color=C_ANNOT, lw=1.1))
    ax.text((earlier + cross_inv) / 2, y_brack - 0.13,
            "Both earlier — threshold can't tell them apart",
            ha="center", fontsize=10, color=C_ANNOT,
            fontweight="bold", path_effects=stroke())

# Curve labels
ax.text(190, vals_c[190] + 0.14,
        "Climatological", color=C_INV,
        fontsize=11, fontweight="bold", path_effects=stroke())
ax.text(168, phase_vals[168] - 0.22,
        "Phase shift", color=C_PHASE,
        fontsize=11, fontweight="bold", path_effects=stroke())
ax.text(210, amp_vals[210] + 0.10,
        "Amplitude ↓", color=C_AMP,
        fontsize=11, fontweight="bold", path_effects=stroke())

fig.savefig(os.path.join(OUTPUT_DIR, "apac_concept_fig4_both.png"))
plt.close(fig)
print("Fig 4 saved")

print(f"\n=== All figures saved to {OUTPUT_DIR} ===")