"""
2016 vs 2023 case study — what actually happened, sector by sector.

2016 and 2023 are the two most anomalous years in the record but for
different reasons. 2016 was primarily a phase event (ice advanced late),
2023 was primarily an amplitude collapse. This figure puts them side by side
so that contrast is immediately visible.

Three panels saved separately so they can be used independently in the thesis:
    a. 2016 alone — phase and amplitude z-scores by sector
    b. 2023 alone — same layout
    c. 2016 vs 2023 combined — 2x2 grid, the main thesis figure

Z-scores are relative to the pre-2016 standard deviation so phase (days) and
amplitude (Mkm²) are directly comparable in height. Raw values annotated on
each bar so the reader can convert back.

Needs annual_params.csv — specifically the z-score columns computed in
fig_phase_amplitude_timeseries.py (or recomputed here from the same baseline).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ch3_style import (
    apply_style,
    SECTORS_NO_CIRC, SECTOR_COLORS, SECTOR_LABELS,
    stroke, sigma_lines, save_fig, DEFAULT_OUTPUT_DIR,
)

apply_style()

DATA_DIR   = "/user/geog/falejandraperez/sea-ice-phase/scripts/R/Ch3/data"
OUTPUT_DIR = DEFAULT_OUTPUT_DIR
ANNUAL_CSV = os.path.join(DATA_DIR, "annual_params.csv")

# --- Load and compute baselines --------------------------------------------
# Recomputing rather than assuming the z-score columns exist, so this script
# can be run standalone without depending on another script having run first.

print("Loading data...")
annual = pd.read_csv(ANNUAL_CSV, parse_dates=["min_date", "max_date"])

for col in ["max_doy_raw", "min_doy_raw", "amplitude_raw_yr"]:
    if col in annual.columns:
        annual[col] = pd.to_numeric(annual[col], errors="coerce")

mask_bl = annual["Year"].between(1979, 2015)

bl_doy_median = annual[mask_bl].groupby("sector")["max_doy_raw"].median()
bl_amp_median = annual[mask_bl].groupby("sector")["amplitude_raw_yr"].median()
bl_doy_std    = annual[mask_bl].groupby("sector")["max_doy_raw"].std()
bl_amp_std    = annual[mask_bl].groupby("sector")["amplitude_raw_yr"].std()

annual["max_doy_raw_anom_2015"]     = (annual["max_doy_raw"]
                                       - annual["sector"].map(bl_doy_median))
annual["amplitude_raw_anom_2015"]   = (annual["amplitude_raw_yr"]
                                       - annual["sector"].map(bl_amp_median))
annual["max_doy_raw_anom_2015_z"]   = (annual["max_doy_raw_anom_2015"]
                                       / annual["sector"].map(bl_doy_std))
annual["amplitude_raw_anom_2015_z"] = (annual["amplitude_raw_anom_2015"]
                                       / annual["sector"].map(bl_amp_std))

# Sector order and styling for all case study panels
CASE_SECTORS = SECTORS_NO_CIRC
case_labels  = [SECTOR_LABELS[s] for s in CASE_SECTORS]
case_colors  = [SECTOR_COLORS[s] for s in CASE_SECTORS]


# --- Helper: pull z-score and raw values for a given year -----------------

def get_vals(year, var):
    d = annual[annual["Year"] == year].set_index("sector")
    return [float(d.loc[s, var]) if s in d.index else np.nan
            for s in CASE_SECTORS]


# --- Helper: draw a single case study panel --------------------------------
# Used for both the standalone panels and the 2x2 combined figure.

def draw_case_panel(ax, z_vals, raw_vals, title, ylabel, raw_unit, ylim):
    bars = ax.bar(case_labels, z_vals, color=case_colors,
                  width=0.6, edgecolor="white", zorder=3)

    ax.axhline(0, color="#2C2C2A", lw=0.8, zorder=4)
    sigma_lines(ax, levels=(1,))  # ±1σ reference lines

    for bar, z, raw in zip(bars, z_vals, raw_vals):
        if np.isnan(z):
            continue
        # Positive bars: annotate above. Negative bars: annotate inside top.
        ypos = (bar.get_height() + ylim * 0.03 if z >= 0
                else bar.get_height() - ylim * 0.10)
        ax.text(
            bar.get_x() + bar.get_width() / 2, ypos,
            f"{z:+.1f}σ\n({raw:+.0f}{raw_unit})",
            ha="center", va="bottom",
            fontsize=9, fontweight="bold",
            color="#2C2C2A", path_effects=stroke()
        )

    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_ylim(-ylim, ylim)
    ax.tick_params(axis="x", rotation=20, labelsize=10)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    ax.tick_params(axis="y", labelsize=10)

    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11, labelpad=8)


# --- Panels a & b — individual year figures --------------------------------

def plot_single_year(year, suptitle, outfile):
    phase_z   = get_vals(year, "max_doy_raw_anom_2015_z")
    amp_z     = get_vals(year, "amplitude_raw_anom_2015_z")
    phase_raw = get_vals(year, "max_doy_raw_anom_2015")
    amp_raw   = get_vals(year, "amplitude_raw_anom_2015")

    phase_ylim = max(abs(v) for v in phase_z if not np.isnan(v)) * 1.5
    amp_ylim   = max(abs(v) for v in amp_z   if not np.isnan(v)) * 1.5

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    draw_case_panel(
        axes[0], phase_z, phase_raw,
        title    = f"Phase anomaly — {year}",
        ylabel   = "Standard deviations from pre-2016 mean\n(negative = ahead of phase)",
        raw_unit = "d",
        ylim     = phase_ylim,
    )
    draw_case_panel(
        axes[1], amp_z, amp_raw,
        title    = f"Amplitude anomaly — {year}",
        ylabel   = "",
        raw_unit = " Mkm²",
        ylim     = amp_ylim,
    )

    fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, outfile, OUTPUT_DIR)


print("Panel a: 2016 case study")
plot_single_year(
    year     = 2016,
    suptitle = ("2016 — Anomalous Decay: Phase or Amplitude?\n"
                "Z-scored anomalies by sector  |  pre-2016 baseline"),
    outfile  = "fig_case_study_2016_2023_a_2016.png",
)

print("Panel b: 2023 case study")
plot_single_year(
    year     = 2023,
    suptitle = ("2023 — Record Minimum\n"
                "Z-scored anomalies by sector  |  pre-2016 baseline"),
    outfile  = "fig_case_study_2016_2023_b_2023.png",
)


# --- Panel c — 2016 vs 2023 combined (the main thesis figure) --------------
# 2x2 grid: rows = phase / amplitude, columns = 2016 / 2023.
# Shared y-axis per row so the magnitudes are directly comparable across years.
# Year labels annotated above each column rather than in individual titles
# to keep the layout clean.

print("Panel c: 2016 vs 2023 combined")

phase_2016_z   = get_vals(2016, "max_doy_raw_anom_2015_z")
phase_2023_z   = get_vals(2023, "max_doy_raw_anom_2015_z")
amp_2016_z     = get_vals(2016, "amplitude_raw_anom_2015_z")
amp_2023_z     = get_vals(2023, "amplitude_raw_anom_2015_z")
phase_2016_raw = get_vals(2016, "max_doy_raw_anom_2015")
phase_2023_raw = get_vals(2023, "max_doy_raw_anom_2015")
amp_2016_raw   = get_vals(2016, "amplitude_raw_anom_2015")
amp_2023_raw   = get_vals(2023, "amplitude_raw_anom_2015")

# Shared ylim per row so 2016 and 2023 are on the same scale
phase_ylim = max(
    max(abs(v) for v in phase_2016_z if not np.isnan(v)),
    max(abs(v) for v in phase_2023_z if not np.isnan(v))
) * 1.5

amp_ylim = max(
    max(abs(v) for v in amp_2016_z if not np.isnan(v)),
    max(abs(v) for v in amp_2023_z if not np.isnan(v))
) * 1.5

fig, axes = plt.subplots(2, 2, figsize=(16, 10),
                         sharex=False, sharey="row")

panel_data = [
    (axes[0, 0], phase_2016_z, phase_2016_raw,
     "Phase anomaly — 2016",
     "Standard deviations\n(negative = ahead of phase)",
     "d", phase_ylim),
    (axes[0, 1], phase_2023_z, phase_2023_raw,
     "Phase anomaly — 2023",
     "", "d", phase_ylim),
    (axes[1, 0], amp_2016_z, amp_2016_raw,
     "Amplitude anomaly — 2016",
     "Standard deviations\n(negative = smaller cycle)",
     " Mkm²", amp_ylim),
    (axes[1, 1], amp_2023_z, amp_2023_raw,
     "Amplitude anomaly — 2023",
     "", " Mkm²", amp_ylim),
]

for ax, z_vals, raw_vals, title, ylabel, raw_unit, ylim in panel_data:
    draw_case_panel(ax, z_vals, raw_vals, title, ylabel, raw_unit, ylim)

# Year labels above each column — cleaner than repeating the year in every title
axes[0, 0].annotate("2016", xy=(0.5, 1.08), xycoords="axes fraction",
                    ha="center", fontsize=14, fontweight="bold",
                    color="#D85A30")
axes[0, 1].annotate("2023", xy=(0.5, 1.08), xycoords="axes fraction",
                    ha="center", fontsize=14, fontweight="bold",
                    color="#BA7517")

fig.suptitle(
    "2016 vs 2023 — Phase and Amplitude Anomalies by Sector\n"
    "Z-scored relative to pre-2016 standard deviation",
    fontsize=13, fontweight="bold", y=1.01
)
fig.tight_layout()
save_fig(fig, "fig_case_study_2016_2023_c_combined.png", OUTPUT_DIR)

print("\nAll case study panels complete.")