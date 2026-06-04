"""
fig01_concept_manuscript.py
2x2 manuscript figure illustrating APAC phase vs amplitude decomposition.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import os

# =============================================================================
# SETTINGS
# =============================================================================

DAILY_CSV  = "/user/geog/falejandraperez/sea-ice-phase/scripts/R/Ch3/data/daily_fitted.csv"
OUTPUT_DIR = "/user/geog/falejandraperez/sea-ice-phase/scripts/python/plotting/Ch3/figures/"
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
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "white",
})

C_INV   = "#2C2C2A"
C_PHASE = "#D4537E"
C_AMP   = "#1D9E75"
C_ANNOT = "#5F5E5A"

PHASE_SHIFT = 12
AMP_CHANGE  = -0.18

MONTH_DAYS   = [0, 28, 59, 89, 120, 150, 181, 212, 242, 273, 303, 334]
MONTH_LABELS = ["Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec","Jan"]

# =============================================================================
# BUILD INVARIANT CYCLE
# =============================================================================

daily      = pd.read_csv(DAILY_CSV)
inv_by_doy = daily.groupby("DOY")["fitted_invariant"].mean()
min_doy    = int(inv_by_doy.idxmin())

doys   = np.arange(1, 366)
vals   = np.array([inv_by_doy[d] for d in doys])
shift  = min_doy - 1
vals_c = gaussian_filter1d(np.roll(vals, -shift), sigma=2)
days   = np.arange(365)

# Synthetic curves
phase_vals = np.roll(vals_c, -PHASE_SHIFT)
inv_min    = vals_c.min()
inv_range  = vals_c.max() - inv_min
amp_vals   = inv_min + (vals_c - inv_min) * ((inv_range + AMP_CHANGE) / inv_range)

def find_right_min(v, start=280):
    return start + int(np.argmin(v[start:]))

# =============================================================================
# 2x2 FIGURE
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
axes = axes.flatten()

YMIN, YMAX = 0.3, 4.3
panel_labels = ["(a)", "(b)", "(c)", "(d)"]
titles = [
    "Invariant annual cycle",
    "Phase shift — earlier timing, same magnitude",
    "Amplitude change — smaller cycle, same timing",
    "Same summer minimum — different mechanisms",
]

for ax, label, title in zip(axes, panel_labels, titles):
    ax.set_facecolor("white")
    ax.set_xlim(-5, 369)
    ax.set_ylim(YMIN, YMAX)
    ax.set_xticks(MONTH_DAYS)
    ax.set_xticklabels(MONTH_LABELS, fontsize=10)
    ax.tick_params(labelsize=10)
    ax.set_yticks(np.arange(0.5, YMAX, 0.5))
    ax.set_yticklabels([f"{y:.1f}" for y in np.arange(0.5, YMAX, 0.5)], fontsize=10)
    ax.text(0.02, 0.97, label, transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top")
    ax.set_title(title, fontsize=11, pad=8, loc="left", color=C_ANNOT)

# Y labels
axes[0].set_ylabel("SIE (million km²)", fontsize=11)
axes[2].set_ylabel("SIE (million km²)", fontsize=11)
axes[2].set_xlabel("Month  (day 0 = annual minimum, late February)", fontsize=10, color=C_ANNOT)
axes[3].set_xlabel("Month  (day 0 = annual minimum, late February)", fontsize=10, color=C_ANNOT)

# --- Panel A: Invariant cycle ---
ax = axes[0]
ax.plot(days, vals_c, color=C_INV, lw=2.5, zorder=4)
ax.fill_between(days, vals_c, YMIN, color=C_INV, alpha=0.07)

# --- Panel B: Phase shift ---
ax = axes[1]
ax.plot(days, vals_c,     color=C_INV,   lw=2.0, alpha=0.5, zorder=4, label="Invariant cycle")
ax.plot(days, phase_vals, color=C_PHASE, lw=2.5, zorder=5, linestyle="--", label="Phase shifted")
peak_inv   = int(np.argmax(vals_c))
peak_phase = int(np.argmax(phase_vals))
ax.axvline(peak_inv,   color=C_INV,   lw=1.0, linestyle=":", alpha=0.5)
ax.axvline(peak_phase, color=C_PHASE, lw=1.0, linestyle=":", alpha=0.5)
ax.legend(fontsize=9, frameon=False, loc="upper left")

# --- Panel C: Amplitude change ---
ax = axes[2]
ax.plot(days, vals_c,   color=C_INV, lw=2.0, alpha=0.5, zorder=4, label="Invariant cycle")
ax.plot(days, amp_vals, color=C_AMP, lw=2.5, zorder=5, linestyle="--", label="Reduced amplitude")
peak_inv = int(np.argmax(vals_c))
ax.axvline(peak_inv, color=C_ANNOT, lw=0.8, linestyle=":", alpha=0.5)
ax.legend(fontsize=9, frameon=False, loc="upper left")

# --- Panel D: Both — same minimum, different mechanisms ---
ax = axes[3]
ax.plot(days, vals_c,     color=C_INV,   lw=2.0, alpha=0.5, zorder=4, label="Invariant cycle")
ax.plot(days, phase_vals, color=C_PHASE, lw=2.5, zorder=5, linestyle="--", label="Phase shifted")
ax.plot(days, amp_vals,   color=C_AMP,   lw=2.5, zorder=5, linestyle="--", label="Reduced amplitude")

# Mark the summer minima
for v, color in [(vals_c, C_INV), (phase_vals, C_PHASE), (amp_vals, C_AMP)]:
    d = find_right_min(v)
    ax.scatter(d, v[d], color=color, s=70, zorder=7,
               edgecolors="white", linewidth=1.5)

# Add annotation showing minima converge
min_phase = find_right_min(phase_vals)
min_amp   = find_right_min(amp_vals)
ax.annotate("",
    xy=(min_phase, phase_vals[min_phase] + 0.05),
    xytext=(min_amp, amp_vals[min_amp] + 0.05),
    arrowprops=dict(arrowstyle="<->", color=C_ANNOT, lw=1.2),
)
ax.text((min_phase + min_amp) / 2, max(phase_vals[min_phase], amp_vals[min_amp]) + 0.12,
        "Similar\nminimum SIE", ha="center", fontsize=8, color=C_ANNOT)

ax.legend(fontsize=9, frameon=False, loc="upper left")

# Overall title
fig.suptitle(
    "Figure 1. The traditional SIE anomaly conflates timing and magnitude",
    fontsize=13, fontweight="bold", y=1.01, x=0.02, ha="left"
)

fig.tight_layout()
fig.subplots_adjust(hspace=0.35, wspace=0.15)
fig.savefig(os.path.join(OUTPUT_DIR, "fig01_concept_manuscript.png"))
plt.close(fig)
print(f"Manuscript figure saved to {OUTPUT_DIR}fig1_concept_manuscript.png")