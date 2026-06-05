"""
fig_concept_manuscript.py
2x2 manuscript figure illustrating APAC phase vs amplitude decomposition.
Follows same structure and paths as Plot_ch3_figures.py
"""

import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# =============================================================================
# PATHS
# =============================================================================

DATA_DIR    = "/user/geog/falejandraperez/sea-ice-phase/scripts/R/Ch3/data"
OUTPUT_DIR  = "/user/geog/falejandraperez/sea-ice-phase/scripts/R/Ch3/figures"
GDRIVE_DEST = "gdrive:results/Ch3_Figures/"
DAILY_CSV   = os.path.join(DATA_DIR, "daily_fitted.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# STYLE
# =============================================================================

plt.rcParams.update({
    "font.family"      : "Nimbus Sans",
    "font.size"        : 11,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.linewidth"   : 0.8,
    "axes.labelsize"   : 12,
    "axes.titlesize"   : 12,
    "axes.titleweight" : "normal",
    "xtick.labelsize"  : 10,
    "ytick.labelsize"  : 10,
    "legend.fontsize"  : 9,
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

PHASE_SHIFT = 15
AMP_CHANGE  = -0.6

MONTH_DAYS   = [0, 28, 59, 89, 120, 150, 181, 212, 242, 273, 303, 334]
MONTH_LABELS = ["Feb","Mar","Apr","May","Jun","Jul",
                "Aug","Sep","Oct","Nov","Dec","Jan"]

# =============================================================================
# BUILD INVARIANT CYCLE CENTRED ON MINIMUM
# =============================================================================

print("Loading data...")
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
# 2x2 MANUSCRIPT FIGURE
# =============================================================================

print("Building 2x2 concept figure...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
axes = axes.flatten()

YMIN, YMAX = 0.3, 4.6
panel_labels = ["(a)", "(b)", "(c)", "(d)"]
titles = [
    "Invariant annual cycle",
    "Phase shift — earlier timing, same magnitude",
    "Amplitude change — smaller cycle, same timing",
    "Same summer minimum — different mechanisms",
]

for ax, label, title in zip(axes, panel_labels, titles):
    ax.set_facecolor("white")
    ax.set_xlim(-5, 364)
    ax.set_ylim(YMIN, YMAX)
    ax.set_xticks(MONTH_DAYS)
    ax.set_xticklabels(MONTH_LABELS, fontsize=10)
    ax.tick_params(labelsize=10)
    ax.set_yticks(np.arange(0.5, YMAX, 0.5))
    ax.set_yticklabels(
        [f"{y:.1f}" for y in np.arange(0.5, YMAX, 0.5)], fontsize=10)
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
peak_inv   = int(np.argmax(vals_c))
peak_phase = int(np.argmax(phase_vals))
ax.axvline(peak_inv,   color=C_INV,   lw=1.0, linestyle=":", alpha=0.4)
ax.axvline(peak_phase, color=C_PHASE, lw=1.0, linestyle=":", alpha=0.6)
ax.legend(fontsize=9, frameon=False, loc="lower left")

# --- Panel C: Amplitude change ---
ax = axes[2]
ax.plot(days, vals_c,   color=C_INV, lw=2.0, alpha=0.5,
        zorder=4, label="Invariant cycle")
ax.plot(days, amp_vals, color=C_AMP, lw=2.5, zorder=5,
        linestyle="--", label="Reduced amplitude")
peak_inv = int(np.argmax(vals_c))
ax.axvline(peak_inv, color=C_ANNOT, lw=0.8, linestyle=":", alpha=0.4)
ax.legend(fontsize=9, frameon=False, loc="lower left")

# --- Panel D: Both — same minimum, different mechanisms ---
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
    ax.scatter(d, v[d], color=color, s=70, zorder=7,
               edgecolors="white", linewidth=1.5)

# Annotate convergence
min_phase = find_right_min(phase_vals)
min_amp   = find_right_min(amp_vals)
mid_x     = (min_phase + min_amp) / 2
mid_y     = max(phase_vals[min_phase], amp_vals[min_amp]) + 0.25
ax.annotate("",
    xy=(min_phase, phase_vals[min_phase] + 0.05),
    xytext=(min_amp, amp_vals[min_amp] + 0.05),
    arrowprops=dict(arrowstyle="<->", color=C_ANNOT, lw=1.2),
)
ax.text(mid_x, mid_y, "Similar minimum SIE",
        ha="center", fontsize=8.5, color=C_ANNOT, style="italic")

ax.legend(fontsize=9, frameon=False, loc="lower left")

fig.tight_layout()
fig.subplots_adjust(hspace=0.35, wspace=0.18)

outpath = os.path.join(OUTPUT_DIR, "fig01_concept_manuscript.png")
fig.savefig(outpath)
plt.close(fig)
print(f"  -> fig01_concept_manuscript.png saved")

# =============================================================================
# SYNC TO GOOGLE DRIVE
# =============================================================================

print(f"\nSyncing to {GDRIVE_DEST}")
result = subprocess.run(
    ["rclone", "copy", outpath, GDRIVE_DEST],
    capture_output=True, text=True
)
if result.returncode == 0:
    print("  ✓ fig01_concept_manuscript.png")
else:
    print(f"  ✗ fig01_concept_manuscript.png: {result.stderr.strip()}")

print("Done.")