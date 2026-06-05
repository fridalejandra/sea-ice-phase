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

PHASE_SHIFT = 15  # days earlier
AMP_SCALE   = 0.78  # fraction of original range (so ~22% smaller)

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

fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
axes = axes.flatten()

YMIN = vals_c.min() - 0.1
YMAX = vals_c.max() + 0.3
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
    yticks = np.arange(np.ceil(YMIN * 2) / 2, YMAX, 0.5)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{y:.1f}" for y in yticks], fontsize=10)
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

# Annotate the shift
ax.annotate("",
    xy=(peak_phase, vals_c.max() * 0.97),
    xytext=(peak_inv, vals_c.max() * 0.97),
    arrowprops=dict(arrowstyle="<->", color=C_ANNOT, lw=1.1)
)
ax.text((peak_phase + peak_inv) / 2, vals_c.max() * 0.99,
        f"~{PHASE_SHIFT} days\nearlier",
        ha="center", va="bottom", fontsize=8, color=C_ANNOT)

ax.legend(fontsize=9, frameon=False, loc="lower left")

# --- Panel C: Amplitude change ---
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
    ax.scatter(d, v[d], color=color, s=80, zorder=7,
               edgecolors="white", linewidth=1.5)

# Annotate convergence of minima
min_inv   = find_right_min(vals_c)
min_phase = find_right_min(phase_vals)
min_amp   = find_right_min(amp_vals)

# horizontal brace showing all three minima are similar
y_annot = min(vals_c[min_inv], phase_vals[min_phase], amp_vals[min_amp]) - 0.18
ax.annotate("",
    xy=(min_phase, y_annot),
    xytext=(min_amp, y_annot),
    arrowprops=dict(arrowstyle="<->", color=C_ANNOT, lw=1.1)
)
ax.text((min_phase + min_amp) / 2, y_annot - 0.15,
        "Similar minimum SIE\ndespite different mechanisms",
        ha="center", va="top", fontsize=8, color=C_ANNOT, style="italic")

ax.legend(fontsize=9, frameon=False, loc="upper left")

fig.tight_layout()
fig.subplots_adjust(hspace=0.38, wspace=0.18)

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