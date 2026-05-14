"""
The conflation argument — why treating SIE anomaly as a single number is misleading.

The traditional SIE anomaly (Extent - climatology) mixes two physically distinct
signals: when the ice peaks (phase) and how large the peak is (amplitude). This
figure shows that directly using two contrasting sectors and years.

EA 2016 is the clearest example: the ice peaked 45 days early, so around the
climatological peak date the raw anomaly looks small or even near-zero — but
that's not because nothing happened. The phase component tells a completely
different story to the amplitude component.

Weddell adds a subtler point: in 2016 the raw anomaly is actually slightly
positive even though phase advanced, because a negative amplitude component
and a positive phase contribution partially cancel. The anomaly changes sign
depending on which component you're looking at.

Layout: 2 rows (EA, Weddell) × 3 columns (raw anomaly | APAC decomposition | components separated)

Needs daily_fitted.csv — specifically:
    fitted_invariant    : climatological mean curve (same for all years at each DOY)
    fitted_apac         : year-specific APAC fitted curve
    raw_anomaly         : Extent - fitted_invariant (what traditional analysis sees)
    amplitude_component : contribution from amplitude shift alone
    phase_component     : contribution from phase shift alone
    trend_component     : long-term trend component
    Extent              : observed daily SIE
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from ch3_style import (
    apply_style,
    SECTOR_COLORS, SECTOR_LABELS,
    zero_line, vline2016, stroke,
    save_fig, DEFAULT_OUTPUT_DIR,
)

apply_style()

DATA_DIR    = "/user/geog/falejandraperez/sea-ice-phase/scripts/R/Ch3/data"
DAILY_CSV   = os.path.join(DATA_DIR, "daily_fitted.csv")
OUTPUT_DIR  = DEFAULT_OUTPUT_DIR

# --- Load ------------------------------------------------------------------

print("Loading daily fitted data...")
daily = pd.read_csv(DAILY_CSV, parse_dates=["Date"])
print(f"  {len(daily)} rows | {daily['Year'].min()}–{daily['Year'].max()}")

# --- Configuration ---------------------------------------------------------
# These two sector/year pairs were chosen because they tell complementary stories.
# EA 2016: phase-dominated anomaly, raw signal is misleadingly small
# Weddell 2016: components partially cancel, raw anomaly even flips sign

PANELS = [
    {
        "sector"    : "SIE_East_Antarctica",
        "year"      : 2016,
        "clim_peak" : 274,   # DOY of climatological peak for this sector
        "note"      : "Phase advance of ~45 days — ice peaked well before\n"
                      "the climatological window. Raw anomaly near zero.",
    },
    {
        "sector"    : "SIE_Weddell",
        "year"      : 2016,
        "clim_peak" : 242,
        "note"      : "Amplitude and phase components partially cancel —\n"
                      "raw anomaly is slightly positive despite a phase advance.",
    },
]

# DOY window to display — showing the full annual cycle
DOY_MIN, DOY_MAX = 1, 366

# Shading around the climatological peak to guide the eye
PEAK_WINDOW = 30   # ± days around clim_peak


# --- Panel drawing functions -----------------------------------------------

def draw_raw_anomaly(ax, data_yr, clim_peak, color, sector, year):
    """
    Left panel: raw SIE anomaly (Extent - climatology).
    This is what a traditional analysis sees — a single integrated signal.
    """
    doy  = data_yr["DOY"].values
    anom = data_yr["raw_anomaly"].values

    ax.fill_between(doy, anom, 0,
                    where=anom >= 0, color="#378ADD", alpha=0.6,
                    linewidth=0, zorder=3, label="Positive anomaly")
    ax.fill_between(doy, anom, 0,
                    where=anom < 0, color="#D4537E", alpha=0.6,
                    linewidth=0, zorder=3, label="Negative anomaly")
    ax.plot(doy, anom, color="#2C2C2A", lw=0.8, zorder=4)

    zero_line(ax)

    # Mark the climatological peak window
    ax.axvspan(clim_peak - PEAK_WINDOW, clim_peak + PEAK_WINDOW,
               color="#FF9800", alpha=0.10, zorder=1)
    ax.axvline(clim_peak, color="#FF9800", lw=1.0, ls=":", zorder=2,
               label=f"Clim. peak (DOY {clim_peak})")

    ax.set_xlim(DOY_MIN, DOY_MAX)
    ax.set_ylabel("SIE anomaly (million km²)", fontsize=10)
    ax.set_title("Traditional SIE anomaly\n(Extent − climatology)",
                 fontsize=10, fontweight="bold")

    # Annotate with sector and year
    ax.text(0.03, 0.96,
            f"{SECTOR_LABELS[sector]}  {year}",
            transform=ax.transAxes, fontsize=10, fontweight="bold",
            va="top", color=color, path_effects=stroke())


def draw_apac_decomposition(ax, data_yr, clim_peak, color):
    """
    Middle panel: APAC fitted curve vs climatological curve.
    The phase shift becomes visible as a horizontal offset between the two peaks.
    """
    doy      = data_yr["DOY"].values
    observed = data_yr["Extent"].values
    clim     = data_yr["fitted_invariant"].values
    fitted   = data_yr["fitted_apac"].values

    # Observed (faint)
    ax.plot(doy, observed, color="#B4B2A9", lw=1.0,
            zorder=2, alpha=0.6, label="Observed")

    # Climatology
    ax.plot(doy, clim, color="#2C2C2A", lw=1.8,
            ls="--", zorder=3, label="Climatology (invariant)")

    # APAC fit
    ax.plot(doy, fitted, color=color, lw=2.2,
            zorder=4, label="APAC fitted")

    # Mark both peaks
    clim_peak_val  = clim[np.argmax(clim)]
    fitted_peak_doy = doy[np.argmax(fitted)]
    fitted_peak_val = fitted[np.argmax(fitted)]

    ax.scatter([clim_peak], [clim_peak_val],
               color="#2C2C2A", s=60, zorder=5)
    ax.scatter([fitted_peak_doy], [fitted_peak_val],
               color=color, s=60, zorder=5)

    # Arrow showing the phase shift
    phase_shift = fitted_peak_doy - clim_peak
    ax.annotate(
        f"Phase shift\n{phase_shift:+d} days",
        xy=(fitted_peak_doy, fitted_peak_val),
        xytext=(fitted_peak_doy + 25, fitted_peak_val + 0.15),
        fontsize=8, color=color,
        arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
        path_effects=stroke(lw=3),
    )

    # Climatological peak window
    ax.axvspan(clim_peak - PEAK_WINDOW, clim_peak + PEAK_WINDOW,
               color="#FF9800", alpha=0.10, zorder=1)
    ax.axvline(clim_peak, color="#FF9800", lw=1.0, ls=":", zorder=2)

    ax.set_xlim(DOY_MIN, DOY_MAX)
    ax.set_ylabel("SIE (million km²)", fontsize=10)
    ax.set_title("APAC fitted curve vs climatology\n(phase shift visible as peak offset)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")


def draw_components(ax, data_yr, clim_peak, color):
    """
    Right panel: amplitude and phase components shown as separate curves.
    This is the key panel — it shows which signal is actually driving the anomaly
    and makes clear they can work in opposite directions.
    """
    doy       = data_yr["DOY"].values
    raw_anom  = data_yr["raw_anomaly"].values
    amp_comp  = data_yr["amplitude_component"].values
    phase_comp= data_yr["phase_component"].values
    trend_comp= data_yr["trend_component"].values

    # Raw anomaly in grey for reference
    ax.plot(doy, raw_anom, color="#B4B2A9", lw=1.2,
            ls="-", zorder=2, alpha=0.7, label="Raw anomaly (reference)")

    # Amplitude component
    ax.plot(doy, amp_comp, color="#378ADD", lw=2.0,
            zorder=4, label="Amplitude component")

    # Phase component
    ax.plot(doy, phase_comp, color="#D4537E", lw=2.0,
            zorder=4, label="Phase component")

    # Trend component (usually small but honest to show it)
    ax.plot(doy, trend_comp, color="#888780", lw=1.2,
            ls="--", zorder=3, alpha=0.7, label="Trend component")

    zero_line(ax)

    # Climatological peak window
    ax.axvspan(clim_peak - PEAK_WINDOW, clim_peak + PEAK_WINDOW,
               color="#FF9800", alpha=0.10, zorder=1)
    ax.axvline(clim_peak, color="#FF9800", lw=1.0, ls=":", zorder=2)

    ax.set_xlim(DOY_MIN, DOY_MAX)
    ax.set_ylabel("SIE anomaly (million km²)", fontsize=10)
    ax.set_title("Decomposed components\n(amplitude, phase, trend)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")


# --- Build the figure ------------------------------------------------------
# 2 rows × 3 columns. Shared x-axis within each row.

fig, axes = plt.subplots(
    2, 3, figsize=(17, 9),
    sharex=True, sharey=False,
)

for row, cfg in enumerate(PANELS):
    sector    = cfg["sector"]
    year      = cfg["year"]
    clim_peak = cfg["clim_peak"]
    color     = SECTOR_COLORS[sector]

    data_yr = (daily[(daily["sector"] == sector) & (daily["Year"] == year)]
               .sort_values("DOY").reset_index(drop=True))

    draw_raw_anomaly(axes[row, 0], data_yr, clim_peak, color, sector, year)
    draw_apac_decomposition(axes[row, 1], data_yr, clim_peak, color)
    draw_components(axes[row, 2], data_yr, clim_peak, color)

    # Annotation note on the right edge
    axes[row, 2].text(
        1.02, 0.5, cfg["note"],
        transform=axes[row, 2].transAxes,
        fontsize=8, va="center", ha="left",
        color="#5F5E5A", style="italic",
        wrap=True,
    )

    # DOY x-axis label on bottom row only
    for ax in axes[row]:
        if row == len(PANELS) - 1:
            ax.set_xlabel("Day of Year", fontsize=10)

# Shared orange legend entry for the peak window — added once outside the loop
peak_patch = mpatches.Patch(color="#FF9800", alpha=0.3,
                             label="Climatological peak window (±30 days)")
fig.legend(handles=[peak_patch], loc="lower center",
           fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.01))

fig.suptitle(
    "The conflation problem: SIE anomaly mixes phase and amplitude signals\n"
    "East Antarctica and Weddell, 2016",
    fontsize=13, fontweight="bold", y=1.01,
)

fig.tight_layout(rect=[0, 0.03, 0.95, 1])
save_fig(fig, "fig_conflation_argument.png", OUTPUT_DIR)

print("fig_conflation_argument.png saved.")