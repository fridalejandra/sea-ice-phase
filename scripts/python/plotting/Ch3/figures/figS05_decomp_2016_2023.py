"""
figS05_decomp_2016_2023.py — CORRECTED for Pipeline B
======================================================
APAC decomposition for all 5 sectors in 2016 and 2023.
10 rows × 3 columns.

WHAT CHANGED FROM THE PREVIOUS VERSION

(1) Reads daily_fitted_B.csv. The previous version read the Pipeline A
    components, where phase_component was built from the wrong curve
    (fitted_phase - fitted_amp instead of fitted_apac - fitted_amp) and the
    trend was double-counted. The plotted curves therefore did not sum to the
    anomaly (~9% mean error). They now sum exactly (~1e-19).

(2) Column 1 now plots anomaly_from_iac (Extent - trend-free climatology),
    which is the black line in H&R Fig 7a and the quantity the components
    actually decompose. The old version plotted raw_anomaly
    (Extent - fitted_apac), which is the RESIDUAL, not the anomaly — so the
    left and right panels were showing different quantities.

(3) Column 3 adds the estimated-anomaly line (est_anomaly = ARMA conditional
    mean from the GARCH fit), the orange curve in H&R Fig 7a. The old
    est_anomaly was raw_anomaly - volatility, which subtracted a positive
    quantity (sigma) and was biased by -0.022.

(4) Phase-shift annotation now uses the OBSERVED max-DOY anomaly
    (max_doy_raw_anom from annual_params_B.csv), not argmax(fitted_apac).
    The fitted peak is an artifact dominated by the fixed s(DOY) term
    (r with raw max = 0.02-0.46; in Ross r(fitted max, raw MIN) = -0.74).
    Example: King Haakon 2016 read "+2 days" from the fitted curve; the
    observed max-DOY anomaly is +9.5 days.

(5) Sum check printed per panel; the script refuses to run on Pipeline A data.
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

from ch3_config import DATA_DIR, DAILY_CSV, ANNUAL_CSV, OUTPUT_DIR, GDRIVE

print("Loading Pipeline B data...")
daily  = pd.read_csv(DAILY_CSV, parse_dates=["Date"])
annual = pd.read_csv(ANNUAL_CSV)
print(f"  daily  {len(daily)} rows | {daily['Year'].min()}–{daily['Year'].max()}")

# --- Guard: refuse Pipeline A input ------------------------------------------
need = ["anomaly_from_iac", "trend_component", "amplitude_component",
        "phase_component", "residual_apac", "est_anomaly", "iac_notrend"]
miss = [c for c in need if c not in daily.columns]
if miss:
    raise SystemExit(f"Missing {miss} — run APAC_Sector_Pipeline_B.R first.")

err = np.nanmean(np.abs(
    daily["anomaly_from_iac"]
    - (daily["trend_component"] + daily["amplitude_component"]
       + daily["phase_component"] + daily["residual_apac"])))
print(f"  decomposition sum check: mean |error| = {err:.3e}")
if err > 1e-6:
    raise SystemExit("Components do not sum — this looks like Pipeline A output.")

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


def make_letters(n):
    """(a)...(z) then (aa),(ab)... — 30 panels exceeds 26 letters."""
    import string
    L = string.ascii_lowercase
    out = []
    for i in range(n):
        out.append(f"({L[i]})" if i < 26 else f"({L[i // 26 - 1]}{L[i % 26]})")
    return out


LETTERS_FLAT = make_letters(len(PANELS) * 3)

COL_TITLES = [
    "Anomaly from climatology\n(Extent – trend-free annual cycle)",
    "APAC fitted curve vs climatology\n(observed timing shift annotated)",
    "Decomposed components\n(trend + amplitude + phase + residual)",
]


def add_peak_window(ax, clim_peak):
    ax.axvspan(clim_peak - PEAK_WINDOW, clim_peak + PEAK_WINDOW,
               color="#FF9800", alpha=0.10, zorder=1)
    ax.axvline(clim_peak, color="#FF9800", lw=0.8, ls=":", zorder=2)


def add_letter(ax, letter):
    ax.text(0.03, 0.97, letter, transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top", ha="left", color="#2C2C2A")


def draw_anomaly(ax, data_yr, clim_peak, color, sector, year, letter):
    """Column 1 — anomaly from the trend-free climatology (H&R Fig 7a black)."""
    doy  = data_yr["DOY"].values
    anom = data_yr["anomaly_from_iac"].values          # [CHANGE 2]
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


def draw_apac_fit(ax, data_yr, clim_peak, color, obs_max_anom, letter):
    """Column 2 — fitted curve vs climatology, OBSERVED timing annotation."""
    doy      = data_yr["DOY"].values
    observed = data_yr["Extent"].values
    clim     = data_yr["iac_notrend"].values           # trend-free reference
    fitted   = data_yr["fitted_apac"].values

    ax.plot(doy, observed, color="#B4B2A9", lw=0.9, zorder=2, alpha=0.6,
            label="Observed")
    ax.plot(doy, clim, color="#2C2C2A", lw=1.6, ls="--", zorder=3,
            label="Climatology (trend-free)")
    ax.plot(doy, fitted, color=color, lw=2.0, zorder=4, label="APAC fitted")

    # Observed maximum — the metric actually used in the chapter
    obs_max_doy = int(doy[np.argmax(observed)])
    obs_max_val = float(np.max(observed))
    ax.scatter([obs_max_doy], [obs_max_val], color=color, s=40, zorder=5)
    ax.scatter([clim_peak], [clim[np.argmax(clim)]],
               color="#2C2C2A", s=40, zorder=5)

    yr_range = float(np.nanmax(clim) - np.nanmin(clim))
    text_y = obs_max_val - yr_range * 0.15
    text_x = obs_max_doy + 25 if obs_max_anom < 0 else obs_max_doy - 70
    ax.annotate(                                        # [CHANGE 4]
        f"Observed max\n{obs_max_anom:+.1f} days",
        xy=(obs_max_doy, obs_max_val), xytext=(text_x, text_y),
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
    """Column 3 — H&R Fig 7a style. These now sum to the black line."""
    doy      = data_yr["DOY"].values
    anom     = data_yr["anomaly_from_iac"].values
    amp_c    = data_yr["amplitude_component"].values
    phase_c  = data_yr["phase_component"].values
    trend_c  = data_yr["trend_component"].values
    est_anom = data_yr["est_anomaly"].values

    ax.plot(doy, anom,     color="#B4B2A9", lw=1.0, zorder=2, alpha=0.7,
            label="Anomaly")
    ax.plot(doy, est_anom, color="#E8A33D", lw=1.0, zorder=3, alpha=0.9,
            label="Estimated anomaly")           # [CHANGE 3]
    ax.plot(doy, amp_c,    color="#378ADD", lw=1.8, zorder=4, label="Amplitude")
    ax.plot(doy, phase_c,  color="#D4537E", lw=1.8, zorder=4, label="Phase")
    ax.plot(doy, trend_c,  color="#888780", lw=1.0, ls="--", zorder=3,
            alpha=0.7, label="Trend")

    zero_line(ax)
    add_peak_window(ax, clim_peak)
    ax.set_xlim(DOY_MIN, DOY_MAX)
    ax.set_ylabel("Anomaly\n(million km²)", fontsize=8)
    ax.legend(fontsize=6.5, loc="lower right", handlelength=1.5, ncol=2)
    add_letter(ax, letter)


# ── Build figure ──────────────────────────────────────────────────────────────
n_rows = len(PANELS)
fig, axes = plt.subplots(n_rows, 3, figsize=(12, 3.8 * n_rows),
                         sharex=True, sharey=False)
fig.subplots_adjust(hspace=0.35, wspace=0.32,
                    left=0.08, right=0.97, top=0.97, bottom=0.03)

fig.add_artist(plt.Line2D(
    [0.05, 0.95], [1 - 5 / n_rows, 1 - 5 / n_rows],
    transform=fig.transFigure, color="#CCCCCC", linewidth=1.5, linestyle="--"))

for row, cfg in enumerate(PANELS):
    sector, year, clim_peak = cfg["sector"], cfg["year"], cfg["clim_peak"]
    color = SECTOR_COLORS[sector]

    data_yr = (daily[(daily["sector"] == sector) & (daily["Year"] == year)]
               .sort_values("DOY").reset_index(drop=True))
    if len(data_yr) == 0:
        print(f"  WARNING: no data for {sector} {year}")
        continue

    arow = annual[(annual["sector"] == sector) & (annual["Year"] == year)]
    obs_max_anom = float(arow["max_doy_raw_anom"].iloc[0]) if len(arow) else np.nan

    p_err = np.nanmean(np.abs(
        data_yr["anomaly_from_iac"]
        - (data_yr["trend_component"] + data_yr["amplitude_component"]
           + data_yr["phase_component"] + data_yr["residual_apac"])))
    print(f"  {SECTOR_LABELS[sector]:<18} {year}  "
          f"obs max anom {obs_max_anom:+6.1f} d   sum err {p_err:.2e}")

    b = row * 3
    draw_anomaly(axes[row, 0], data_yr, clim_peak, color,
                 sector, year, LETTERS_FLAT[b])
    draw_apac_fit(axes[row, 1], data_yr, clim_peak, color,
                  obs_max_anom, LETTERS_FLAT[b + 1])
    draw_components(axes[row, 2], data_yr, clim_peak, color,
                    LETTERS_FLAT[b + 2])

    if row == n_rows - 1:
        for ax in axes[row]:
            ax.set_xlabel("Day of Year", fontsize=9)

for ax, title in zip(axes[0], COL_TITLES):
    ax.set_title(title, fontsize=9, fontweight="bold", pad=6)

fig.text(0.01, 1 - 2.5 / n_rows, "2016", fontsize=12, fontweight="bold",
         va="center", ha="left", color="#2C2C2A", rotation=90)
fig.text(0.01, 1 - 7.5 / n_rows, "2023", fontsize=12, fontweight="bold",
         va="center", ha="left", color="#2C2C2A", rotation=90)

peak_patch = mpatches.Patch(color="#FF9800", alpha=0.3,
                            label="Climatological peak window (±30 days)")
fig.legend(handles=[peak_patch], loc="lower center",
           fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 0.005))

from ch3_plot import save
save(fig, "figS05_decomp_2016_2023.png", dpi=150)
print("Done.")
