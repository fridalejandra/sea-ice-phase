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
C_ANNOT  = "#5F5E5A"

PHASE_SHIFT = 12
AMP_CHANGE  = -0.18

# =============================================================================
# 1. BUILD INVARIANT CYCLE CENTRED ON MINIMUM
# =============================================================================

daily      = pd.read_csv(DAILY_CSV)
inv_by_doy = daily.groupby("DOY")["fitted_invariant"].mean()
min_doy    = int(inv_by_doy.idxmin())

doys   = np.arange(1, 366)
vals   = np.array([inv_by_doy[d] for d in doys])
shift  = min_doy - 1
vals_c = gaussian_filter1d(np.roll(vals, -shift), sigma=2)
days   = np.arange(365)

MONTH_DAYS   = [0,  28,  59,  89, 120, 150, 181, 212, 242, 273, 303, 334]
MONTH_LABELS = ["Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec","Jan"]

# =============================================================================
# 2. BUILD SYNTHETIC CURVES
# =============================================================================

phase_vals = np.roll(vals_c, -PHASE_SHIFT)
inv_min    = vals_c.min()
inv_range  = vals_c.max() - inv_min
amp_vals   = inv_min + (vals_c - inv_min) * ((inv_range + AMP_CHANGE) / inv_range)

# =============================================================================
# 3. HELPER FUNCTIONS
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
    yticks = np.arange(0.5, ymax, 0.5)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{y:.1f}" for y in yticks], fontsize=11)


def stroke(fg="white"):
    return [pe.withStroke(linewidth=3, foreground=fg)]


def find_right_min(vals, start=280):
    seg = vals[start:]
    return start + int(np.argmin(seg))


# =============================================================================
# FIG 1 — Invariant cycle only
# =============================================================================

fig, ax = plt.subplots(figsize=(11, 6))
style_ax(ax, "The invariant annual cycle")

ax.plot(days, vals_c, color=C_INV, lw=2.5, zorder=4)
ax.fill_between(days, vals_c, 0.3, color=C_INV, alpha=0.07)

ax.text(185, vals_c[185] + 0.14,
        "Climatological mean cycle", color=C_INV,
        fontsize=11, fontweight="bold", path_effects=stroke())

fig.tight_layout()
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

peak_inv   = int(np.argmax(vals_c))
peak_phase = int(np.argmax(phase_vals))

ax.axvline(peak_inv,   color=C_INV,   lw=1.0, linestyle=":", alpha=0.6, zorder=3)
ax.axvline(peak_phase, color=C_PHASE, lw=1.0, linestyle=":", alpha=0.6, zorder=3)

ax.text(190, vals_c[190] + 0.14,
        "Climatological cycle", color=C_INV,
        fontsize=11, fontweight="bold", path_effects=stroke())
ax.text(163, phase_vals[163] - 0.24,
        "Phase shifted  (−12 days)", color=C_PHASE,
        fontsize=11, fontweight="bold", path_effects=stroke(), va="top")

fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "apac_concept_fig2_phase.png"))
plt.close(fig)
print("Fig 2 saved")

# =============================================================================
# FIG 3 — Amplitude change
# =============================================================================

fig, ax = plt.subplots(figsize=(11, 6))
style_ax(ax, "Amplitude change — smaller cycle, same timing")

ax.plot(days, vals_c,   color=C_INV, lw=2.5, alpha=0.6, zorder=4)
ax.plot(days, amp_vals, color=C_AMP, lw=2.5, zorder=5, linestyle="--")

peak_inv = int(np.argmax(vals_c))
ax.axvline(peak_inv, color=C_ANNOT, lw=0.8, linestyle=":", alpha=0.5)

ax.text(190, vals_c[190] + 0.14,
        "Climatological cycle", color=C_INV,
        fontsize=11, fontweight="bold", path_effects=stroke())
ax.text(190, amp_vals[190] - 0.24,
        "Amplitude reduced  (−0.18 Mkm²)", color=C_AMP,
        fontsize=11, fontweight="bold", path_effects=stroke(), va="top")

fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "apac_concept_fig3_amplitude.png"))
plt.close(fig)
print("Fig 3 saved")

# =============================================================================
# FIG 4 — Same low minimum, different mechanisms
# =============================================================================

fig, ax = plt.subplots(figsize=(11, 6))
style_ax(ax, "Same low minimum — completely different mechanisms")

ax.plot(days, vals_c,     color=C_INV,   lw=2.5, alpha=0.7, zorder=4)
ax.plot(days, phase_vals, color=C_PHASE, lw=2.5, zorder=5, linestyle="--")
ax.plot(days, amp_vals,   color=C_AMP,   lw=2.5, zorder=5, linestyle="--")

min_inv   = find_right_min(vals_c)
min_phase = find_right_min(phase_vals)
min_amp   = find_right_min(amp_vals)

val_inv   = vals_c[min_inv]
val_phase = phase_vals[min_phase]
val_amp   = amp_vals[min_amp]

for day, val, color in [(min_inv,   val_inv,   C_INV),
                        (min_phase, val_phase, C_PHASE),
                        (min_amp,   val_amp,   C_AMP)]:
    ax.plot([day - 18, day + 18], [val, val],
            color=color, lw=1.5, linestyle=":", zorder=6)
    ax.scatter(day, val, color=color, s=60, zorder=7,
               edgecolors="white", linewidth=1.2)

bracket_x = min_inv + 22
ax.annotate("",
            xy=(bracket_x, val_phase),
            xytext=(bracket_x, val_inv),
            arrowprops=dict(arrowstyle="<->", color=C_ANNOT, lw=1.1))
ax.text(bracket_x + 6, (val_inv + val_phase) / 2,
        "Lower minimum\n(same anomaly signal)",
        ha="left", va="center", fontsize=10,
        color=C_ANNOT, fontweight="bold",
        path_effects=stroke())

ax.text(182, 0.42,
        "APAC reads the full curve shape — not just the endpoint.\n"
        "Anomaly analysis sees the same signal. APAC sees why.",
        ha="center", va="bottom", fontsize=10.5,
        color="#2C2C2A", style="italic",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#F1EFE8",
                  edgecolor="#B4B2A9", linewidth=0.8))

ax.text(100, vals_c[100] + 0.12,
        "Climatological", color=C_INV,
        fontsize=11, fontweight="bold", path_effects=stroke())
ax.text(80, phase_vals[80] - 0.20,
        "Phase shift", color=C_PHASE,
        fontsize=11, fontweight="bold", path_effects=stroke(), va="top")
ax.text(120, amp_vals[120] - 0.20,
        "Amplitude ↓", color=C_AMP,
        fontsize=11, fontweight="bold", path_effects=stroke(), va="top")

fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "apac_concept_fig4_both.png"))
plt.close(fig)
print("Fig 4 saved")

print(f"\n=== All figures saved to {OUTPUT_DIR} ===")