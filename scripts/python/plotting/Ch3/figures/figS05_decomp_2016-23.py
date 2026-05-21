"""
figS05_decomp_2016_2023.py
===========================
Supplementary figure: APAC decomposition for all 5 sectors in 2016 and 2023.
10 rows × 3 columns (one row per sector-year combination).
Same panel structure as fig01_conflation.py.

Panels:
  Col 1: Traditional SIE anomaly
  Col 2: APAC fitted curve vs climatology
  Col 3: Decomposed components (amplitude, phase, trend)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import subprocess

from ch3_style import (
    apply_style,
    SECTOR_COLORS, SECTOR_LABELS,
    zero_line, stroke,
    save_fig, DEFAULT_OUTPUT_DIR,
)

apply_style()

DATA_DIR   = "/user/geog/falejandraperez/sea-ice-phase/scripts/R/Ch3/data"
DAILY_CSV  = os.path.join(DATA_DIR, "daily_fitted.csv")
OUTPUT_DIR = DEFAULT_OUTPUT_DIR
GDRIVE     = "gdrive:sea-ice-phase/results/Ch3_Figures/"

print("Loading daily fitted data...")
daily = pd.read_csv(DAILY_CSV, parse_dates=["Date"])
print(f"  {len(daily)} rows | {daily['Year'].min()}–{daily['Year'].max()}")

PANELS = [
    {"sector": "SIE_East_Antarctica",         "year": 2016, "clim_peak": 273},
    {"sector": "SIE_Weddell",                 "year": 2016, "clim_peak": 243},
    {"sector": "SIE_Ross",                    "year": 2016, "clim_peak": 277},
    {"sector": "SIE_Amundsen_Bellingshausen", "year": 2016, "clim_peak": 245},
    {"sector": "SIE_King_Haakon",             "year": 2016, "clim_peak": 269},
    {"sector": "SIE_East_Antarctica",         "year": 2023, "clim_peak": 273},
    {"sector": "SIE_Weddell",                 "year": 2023, "clim_peak": 243},
    {"sector": "SIE_Ross",                    "year": 2023, "clim_peak": 277},
    {"sector": "SIE_Amundsen_Bellingshausen", "year": 2023, "clim_peak": 245},
    {"sector": "SIE_King_Haakon",             "year": 2023, "clim_peak": 269},
]

DOY_MIN, DOY_MAX = 1, 366
PEAK_WINDOW      = 30

# Generate panel letters (a)-(z) then (aa) etc — 30 panels max
def make_letters(n):
    import string
    letters = list(string.ascii_lowercase)
    return [f"({l})" for l in letters[:n]]

LETTERS_FLAT = make_letters(len(PANELS) * 3)

COL_TITLES = [
    "Traditional SIE anomaly\n(Extent – climatology)",
    "APAC fitted curve vs climatology\n(phase shift visible as peak offset)",
    "Decomposed components\n(amplitude, phase, trend)",
]


def add_peak_window(ax, clim_peak):
    ax.axvspan(clim_peak - PEAK_WINDOW, clim_peak + PEAK_WINDOW,
               color="#FF9800", alpha=0.10, zorder=1)
    ax.axvline(clim_peak, color="#FF9800", lw=0.8, ls=":", zorder=2)


def add_letter(ax, letter):
    ax.text(0.03, 0.97, letter, transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top", ha="left",
            color="#2C2C2A")


def draw_raw_anomaly(ax, data_yr, clim_peak, color, sector, year, letter):
    doy  = data_yr["DOY"].values
    anom = data_yr["raw_anomaly"].values
    ax.fill_between(doy, anom, 0, where=anom >= 0,
                    color="#378ADD", alpha=0.55, linewidth=0, zorder=3)
    ax.fill_between(doy, anom, 0, where=anom < 0,
                    color="#D4537E", alpha=0.55, linewidth=0, zorder=3)
    ax.plot(doy, anom, color="#2C2C2A", lw=0.7, zorder=4)
    zero_line(ax)
    add_peak_window(ax, clim_peak)
    ax.set_xlim(DOY_MIN, DOY_MAX)
    ax.set_ylabel("SIE anomaly\n(million km²)", fontsize=8)
    ax.text(0.03, 0.88, f"{SECTOR_LABELS[sector]}  {year}",
            transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="top", color=color, path_effects=stroke())
    add_letter(ax, letter)


def draw_apac_decomposition(ax, data_yr, clim_peak, color, letter):
    doy      = data_yr["DOY"].values
    observed = data_yr["Extent"].values
    clim     = data_yr["fitted_invariant"].values
    fitted   = data_yr["fitted_apac"].values

    ax.plot(doy, observed, color="#B4B2A9", lw=0.9, zorder=2, alpha=0.6,
            label="Observed")
    ax.plot(doy, clim, color="#2C2C2A", lw=1.6, ls="--", zorder=3,
            label="Climatology (invariant)")
    ax.plot(doy, fitted, color=color, lw=2.0, zorder=4, label="APAC fitted")

    clim_peak_val   = clim[np.argmax(clim)]
    fitted_peak_doy = int(doy[np.argmax(fitted)])
    fitted_peak_val = fitted[np.argmax(fitted)]

    ax.scatter([clim_peak],       [clim_peak_val],   color="#2C2C2A", s=40, zorder=5)
    ax.scatter([fitted_peak_doy], [fitted_peak_val], color=color,     s=40, zorder=5)

    phase_shift = fitted_peak_doy - clim_peak
    ax_ymin, ax_ymax = clim.min(), clim.max()
    text_y = fitted_peak_val - (ax_ymax - ax_ymin) * 0.15
    text_x = fitted_peak_doy + 25 if phase_shift < 0 else fitted_peak_doy - 70
    ax.annotate(
        f"Phase shift\n{phase_shift:+d} days",
        xy=(fitted_peak_doy, fitted_peak_val),
        xytext=(text_x, text_y),
        fontsize=7.5, color=color, ha="center",
        arrowprops=dict(arrowstyle="->", color=color, lw=1.0),
        path_effects=stroke(lw=2),
    )

    add_peak_window(ax, clim_peak)
    ax.set_xlim(DOY_MIN, DOY_MAX)
    ax.set_ylabel("SIE (million km²)", fontsize=8)
    ax.legend(fontsize=7, loc="lower right", handlelength=1.5)
    add_letter(ax, letter)


def draw_components(ax, data_yr, clim_peak, color, letter):
    doy        = data_yr["DOY"].values
    raw_anom   = data_yr["raw_anomaly"].values
    amp_comp   = data_yr["amplitude_component"].values
    phase_comp = data_yr["phase_component"].values
    trend_comp = data_yr["trend_component"].values

    ax.plot(doy, raw_anom,   color="#B4B2A9", lw=1.0, zorder=2, alpha=0.7,
            label="Raw anomaly")
    ax.plot(doy, amp_comp,   color="#378ADD", lw=1.8, zorder=4,
            label="Amplitude")
    ax.plot(doy, phase_comp, color="#D4537E", lw=1.8, zorder=4,
            label="Phase")
    ax.plot(doy, trend_comp, color="#888780", lw=1.0, ls="--", zorder=3,
            alpha=0.7, label="Trend")

    zero_line(ax)
    add_peak_window(ax, clim_peak)
    ax.set_xlim(DOY_MIN, DOY_MAX)
    ax.set_ylabel("Anomaly\n(million km²)", fontsize=8)
    ax.legend(fontsize=7, loc="lower right", handlelength=1.5)
    add_letter(ax, letter)


# ── Build figure ──────────────────────────────────────────────────────────────
n_rows = len(PANELS)
fig, axes = plt.subplots(
    n_rows, 3,
    figsize=(12, 3.8 * n_rows),
    sharex=True, sharey=False,
)
fig.subplots_adjust(hspace=0.35, wspace=0.32,
                    left=0.08, right=0.97,
                    top=0.97, bottom=0.03)

# Year divider line between 2016 and 2023 rows
fig.add_artist(plt.Line2D(
    [0.05, 0.95], [1 - 5/n_rows, 1 - 5/n_rows],
    transform=fig.transFigure,
    color="#CCCCCC", linewidth=1.5, linestyle="--"
))

for row, cfg in enumerate(PANELS):
    sector    = cfg["sector"]
    year      = cfg["year"]
    clim_peak = cfg["clim_peak"]
    color     = SECTOR_COLORS[sector]

    data_yr = (daily[(daily["sector"] == sector) & (daily["Year"] == year)]
               .sort_values("DOY").reset_index(drop=True))

    if len(data_yr) == 0:
        print(f"  WARNING: no data for {sector} {year}")
        continue

    letter_base = row * 3
    draw_raw_anomaly(axes[row, 0], data_yr, clim_peak, color,
                     sector, year, LETTERS_FLAT[letter_base])
    draw_apac_decomposition(axes[row, 1], data_yr, clim_peak, color,
                            LETTERS_FLAT[letter_base + 1])
    draw_components(axes[row, 2], data_yr, clim_peak, color,
                    LETTERS_FLAT[letter_base + 2])

    if row == n_rows - 1:
        for ax in axes[row]:
            ax.set_xlabel("Day of Year", fontsize=9)

# Column titles on top row
for ax, title in zip(axes[0], COL_TITLES):
    ax.set_title(title, fontsize=9, fontweight="bold", pad=6)

# Year labels on left margin
fig.text(0.01, 1 - 2.5/n_rows, "2016", fontsize=12, fontweight="bold",
         va="center", ha="left", color="#2C2C2A", rotation=90)
fig.text(0.01, 1 - 7.5/n_rows, "2023", fontsize=12, fontweight="bold",
         va="center", ha="left", color="#2C2C2A", rotation=90)

# Legend
peak_patch = mpatches.Patch(color="#FF9800", alpha=0.3,
                             label="Climatological peak window (±30 days)")
fig.legend(handles=[peak_patch], loc="lower center",
           fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 0.005))

# Save
outpath = os.path.join(OUTPUT_DIR, "figS05_decomp_2016_2023.png")
fig.savefig(outpath, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved → {outpath}")

result = subprocess.run(
    ["rclone", "copy", outpath, GDRIVE],
    capture_output=True, text=True
)
if result.returncode == 0:
    print(f"✓ Synced → {GDRIVE}")
else:
    print(f"✗ rclone failed: {result.stderr.strip()}")

print("Done.")