"""
apac_concept_figures.py
Generates 4 PNG figures illustrating APAC phase vs amplitude decomposition

Fig 1: Invariant annual cycle only
Fig 2: + Phase shift (synthetic, 12 days earlier)
Fig 3: + Amplitude change (synthetic, -0.18 Mkm2 peak reduction)
Fig 4: All three + threshold line with annotations

Output: /user/geog/falejandraperez/sea-ice-phase/scripts/R/chapter3/figures/
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

# Colors — white background Labe style
C_INV    = "#2C2C2A"    # near-black for invariant cycle
C_PHASE  = "#D4537E"    # pink-red for phase shift
C_AMP    = "#1D9E75"    # teal-green for amplitude change
C_THRESH = "#BA7517"    # amber for threshold
BG       = "white"

PHASE_SHIFT = 12        # days earlier
AMP_CHANGE  = -0.18     # million km2 reduction in peak

# =============================================================================
# 1. BUILD INVARIANT CYCLE CENTRED ON MINIMUM
# =============================================================================

daily = pd.read_csv(DAILY_CSV)

# Circumpolar invariant cycle by DOY
inv_by_doy = daily.groupby("DOY")["fitted_invariant"].mean()

# Find minimum DOY
min_doy = int(inv_by_doy.idxmin())

# Recentre: DOY min_doy = day 0
doys   = np.arange(1, 366)
vals   = np.array([inv_by_doy[d] for d in doys])

# Roll so min_doy is at index 0
shift  = min_doy - 1
vals_c = np.roll(vals, -shift)
vals_c = gaussian_filter1d(vals_c, sigma=2)  # light smoothing

days   = np.arange(365)

# Month labels centred on minimum (day 0 = ~Feb 18)
# Each month is ~30.4 days from day 0
month_labels = {
    0:   "Feb",
    28:  "Mar",
    59:  "Apr",
    89:  "May",
    120: "Jun",
    150: "Jul",
    181: "Aug",
    212: "Sep",
    242: "Oct",
    273: "Nov",
    303: "Dec",
    334: "Jan",
}

# =============================================================================
# 2. BUILD SYNTHETIC CURVES
# =============================================================================

# Phase shift: roll the invariant curve PHASE_SHIFT days to the left
phase_vals = np.roll(vals_c, -PHASE_SHIFT)

# Amplitude change: same timing, but scale the deviation from minimum
#   new_val = min + (old_val - min) * scale_factor
inv_min  = vals_c.min()
inv_range = vals_c.max() - inv_min
new_range = inv_range + AMP_CHANGE
scale    = new_range / inv_range
amp_vals = inv_min + (vals_c - inv_min) * scale

# Threshold: 50% of invariant max (visually clean intersection)
thresh_val = inv_min + inv_range * 0.50

# =============================================================================
# 3. SHARED PLOT SETTINGS
# =============================================================================

def make_ax(fig, title=None):
    ax = fig.add_subplot(111)
    ax.set_facecolor(BG)
    fig.patch.set_facecolor(BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#2C2C2A")
    ax.spines["bottom"].set_color("#2C2C2A")
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(axis="both", colors="#2C2C2A", length=4, width=1.0,
                   labelsize=12, pad=6)
    ax.set_xlim(-5, 369)
    ax.set_ylim(0.3, 4.3)
    ax.set_xticks(list(month_labels.keys()))
    ax.set_xticklabels(list(month_labels.values()),
                       fontsize=12, color="#2C2C2A", fontweight="bold")
    ax.set_ylabel("SIE (million km²)", fontsize=13, color="#2C2C2A",
                  fontweight="bold", labelpad=10)
    ax.set_xlabel("Month  (day 0 = annual minimum, late February)",
                  fontsize=11, color="#5F5E5A", labelpad=8)
    ax.yaxis.set_tick_params(labelsize=12)
    if title:
        ax.set_title(title, fontsize=15, fontweight="bold",
                     color="#2C2C2A", pad=14, loc="left")
    return ax


def label_curve(ax, days, vals, text, color, day_pos, va="bottom", offset=0.12):
    """Place a label directly on the curve at day_pos."""
    y = vals[day_pos] + offset if va == "bottom" else vals[day_pos] - offset
    ax.text(day_pos, y, text, color=color, fontsize=12, fontweight="bold",
            va=va, ha="center",
            path_effects=[pe.withStroke(linewidth=3, foreground=BG)])


def add_max_annotation(ax, vals, color, label, offset_x=5, offset_y=0.1):
    """Mark the maximum with a dot and label."""
    peak_day = int(np.argmax(vals))
    peak_val = vals[peak_day]
    ax.scatter(peak_day, peak_val, color=color, s=50, zorder=6)
    ax.annotate(f"{label}\npeak: {peak_val:.2f} Mkm²",
                xy=(peak_day, peak_val),
                xytext=(peak_day + offset_x, peak_val + offset_y),
                fontsize=10, color=color, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=color, lw=0.8),
                path_effects=[pe.withStroke(linewidth=3, foreground=BG)])


# =============================================================================
# FIG 1 — Invariant cycle only
# =============================================================================

fig1, ax1 = plt.subplots(figsize=(11, 6))
make_ax(fig1, title="The invariant annual cycle")
ax1 = fig1.axes[0]

ax1.plot(days, vals_c, color=C_INV, lw=2.8, zorder=4)
ax1.fill_between(days, vals_c, 0.3, color=C_INV, alpha=0.06)

label_curve(ax1, days, vals_c, "Climatological\nmean cycle",
            C_INV, 200, va="bottom", offset=0.15)

ax1.annotate("Annual minimum\n(day 0 ≈ Feb 18)",
             xy=(0, vals_c[0]),
             xytext=(30, vals_c[0] + 0.5),
             fontsize=10, color="#5F5E5A",
             arrowprops=dict(arrowstyle="->", color="#5F5E5A", lw=1.0))

ax1.annotate(f"Annual maximum\n≈ mid-September",
             xy=(np.argmax(vals_c), vals_c.max()),
             xytext=(np.argmax(vals_c) - 50, vals_c.max() + 0.2),
             fontsize=10, color="#5F5E5A",
             arrowprops=dict(arrowstyle="->", color="#5F5E5A", lw=1.0))

fig1.tight_layout()
fig1.savefig(os.path.join(OUTPUT_DIR, "apac_concept_fig1_invariant.png"),
             dpi=200, bbox_inches="tight", facecolor=BG)
plt.close(fig1)
print("Fig 1 saved")

# =============================================================================
# FIG 2 — Invariant + phase shift
# =============================================================================

fig2, ax2 = plt.subplots(figsize=(11, 6))
make_ax(fig2, title="Phase shift — the whole cycle moves earlier")
ax2 = fig2.axes[0]

ax2.plot(days, vals_c,    color=C_INV,   lw=2.5, zorder=4, alpha=0.7,
         label="Invariant cycle")
ax2.plot(days, phase_vals, color=C_PHASE, lw=2.5, zorder=5,
         linestyle="--", label=f"Phase shift (−{PHASE_SHIFT} days)")

# Annotate peak shift
peak_inv   = int(np.argmax(vals_c))
peak_phase = int(np.argmax(phase_vals))
ymax       = vals_c.max()

ax2.annotate("", xy=(peak_phase, ymax + 0.35),
             xytext=(peak_inv, ymax + 0.35),
             arrowprops=dict(arrowstyle="<->", color="#5F5E5A", lw=1.2))
ax2.text((peak_inv + peak_phase) / 2, ymax + 0.45,
         f"−{PHASE_SHIFT} days", ha="center", fontsize=11,
         color="#5F5E5A", fontweight="bold")

ax2.text(peak_inv + 3, ymax + 0.1, "Same peak\nheight",
         ha="left", fontsize=10, color="#5F5E5A",
         path_effects=[pe.withStroke(linewidth=3, foreground=BG)])

label_curve(ax2, days, vals_c,    "Climatological", C_INV,   180, offset= 0.15)
label_curve(ax2, days, phase_vals, "Phase shifted",  C_PHASE, 160, offset=-0.25, va="top")

fig2.tight_layout()
fig2.savefig(os.path.join(OUTPUT_DIR, "apac_concept_fig2_phase.png"),
             dpi=200, bbox_inches="tight", facecolor=BG)
plt.close(fig2)
print("Fig 2 saved")

# =============================================================================
# FIG 3 — Invariant + amplitude change
# =============================================================================

fig3, ax3 = plt.subplots(figsize=(11, 6))
make_ax(fig3, title="Amplitude change — smaller cycle, same timing")
ax3 = fig3.axes[0]

ax3.plot(days, vals_c,   color=C_INV,  lw=2.5, zorder=4, alpha=0.7,
         label="Invariant cycle")
ax3.plot(days, amp_vals, color=C_AMP,  lw=2.5, zorder=5,
         linestyle="--", label="Amplitude reduced")

# Annotate amplitude difference
peak_day   = int(np.argmax(vals_c))
peak_inv_v = vals_c[peak_day]
peak_amp_v = amp_vals[peak_day]

ax3.annotate("", xy=(peak_day + 8, peak_amp_v),
             xytext=(peak_day + 8, peak_inv_v),
             arrowprops=dict(arrowstyle="<->", color="#5F5E5A", lw=1.2))
ax3.text(peak_day + 12, (peak_inv_v + peak_amp_v) / 2,
         f"{AMP_CHANGE:.2f} Mkm²", ha="left", fontsize=11,
         color="#5F5E5A", fontweight="bold",
         path_effects=[pe.withStroke(linewidth=3, foreground=BG)])

ax3.text(peak_day - 40, peak_inv_v + 0.1,
         "Same timing\n(peaks aligned)",
         ha="center", fontsize=10, color="#5F5E5A",
         path_effects=[pe.withStroke(linewidth=3, foreground=BG)])

# Vertical line at peak showing same timing
ax3.axvline(peak_day, color="#5F5E5A", lw=0.8, linestyle=":", alpha=0.6)

label_curve(ax3, days, vals_c,  "Climatological",    C_INV, 180, offset= 0.15)
label_curve(ax3, days, amp_vals, "Amplitude reduced", C_AMP, 180, offset=-0.22, va="top")

fig3.tight_layout()
fig3.savefig(os.path.join(OUTPUT_DIR, "apac_concept_fig3_amplitude.png"),
             dpi=200, bbox_inches="tight", facecolor=BG)
plt.close(fig3)
print("Fig 3 saved")

# =============================================================================
# FIG 4 — All three + threshold + the key insight
# =============================================================================

fig4, ax4 = plt.subplots(figsize=(11, 6))
make_ax(fig4, title="Both cross the threshold earlier — for different reasons")
ax4 = fig4.axes[0]

# Curves
ax4.plot(days, vals_c,    color=C_INV,   lw=2.5, zorder=4, alpha=0.7)
ax4.plot(days, phase_vals, color=C_PHASE, lw=2.5, zorder=5, linestyle="--")
ax4.plot(days, amp_vals,  color=C_AMP,   lw=2.5, zorder=5, linestyle="--")

# Threshold line
ax4.axhline(thresh_val, color=C_THRESH, lw=1.5,
            linestyle=(0, (4, 3)), zorder=3, alpha=0.9)
ax4.text(360, thresh_val + 0.06, "Threshold", color=C_THRESH,
         fontsize=10, fontweight="bold", ha="right",
         path_effects=[pe.withStroke(linewidth=3, foreground=BG)])

# Find threshold crossings on descent (right side, day > 150)
def find_crossing(vals, thresh, start=150):
    for i in range(start, len(vals) - 1):
        if vals[i] >= thresh >= vals[i + 1]:
            return i
    return None

cross_inv   = find_crossing(vals_c,    thresh_val)
cross_phase = find_crossing(phase_vals, thresh_val)
cross_amp   = find_crossing(amp_vals,  thresh_val)

for cross, color in [(cross_inv, C_INV),
                     (cross_phase, C_PHASE),
                     (cross_amp, C_AMP)]:
    if cross:
        ax4.scatter(cross, thresh_val, color=color, s=70,
                    zorder=7, edgecolors="white", linewidth=1.2)

# Bracket showing both cross earlier than invariant
if cross_phase and cross_amp and cross_inv:
    earlier = min(cross_phase, cross_amp)
    y_annot = thresh_val - 0.35
    ax4.annotate("", xy=(earlier, y_annot),
                 xytext=(cross_inv, y_annot),
                 arrowprops=dict(arrowstyle="<->", color="#5F5E5A", lw=1.2))
    ax4.text((earlier + cross_inv) / 2, y_annot - 0.12,
             "Both earlier", ha="center", fontsize=10,
             color="#5F5E5A", fontweight="bold",
             path_effects=[pe.withStroke(linewidth=3, foreground=BG)])
    ax4.text((earlier + cross_inv) / 2, y_annot - 0.28,
             "— threshold can't tell them apart",
             ha="center", fontsize=9, color="#5F5E5A",
             path_effects=[pe.withStroke(linewidth=3, foreground=BG)])

# Labels
label_curve(ax4, days, vals_c,     "Climatological", C_INV,   195, offset= 0.15)
label_curve(ax4, days, phase_vals,  "Phase shift",    C_PHASE, 175, offset=-0.25, va="top")
label_curve(ax4, days, amp_vals,    "Amplitude ↓",    C_AMP,   215, offset= 0.12)

fig4.tight_layout()
fig4.savefig(os.path.join(OUTPUT_DIR, "apac_concept_fig4_both.png"),
             dpi=200, bbox_inches="tight", facecolor=BG)
plt.close(fig4)
print("Fig 4 saved")

print("\n=== All 4 figures saved to", OUTPUT_DIR, "===")