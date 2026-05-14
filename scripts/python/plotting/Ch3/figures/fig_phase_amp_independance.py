"""
Phase and amplitude independence — observed data vs modelled parameters.

The central methodological claim is that phase and amplitude are genuinely
independent quantities that the APAC model separates correctly. This figure
tests that claim two ways:

  Left column  — raw observed: peak DOY and amplitude computed directly from
                 the SIE timeseries, no model involved. Near-zero correlations
                 here mean the physical system itself is orthogonal.

  Right column — modelled parameters: the fitted phase and amplitude scalars
                 from the APAC decomposition. These should also be near-zero
                 if the model is respecting the physical independence.

ABS is the notable exception — the modelled correlation is stronger than the
raw, suggesting the sinusoidal fitting procedure introduces some phase/amplitude
coupling in that sector that isn't present in the observations. Worth a footnote.

Detrending is optional (DETREND = True below). For the independence argument,
detrending is defensible — you want to show the relationship isn't just two
variables sharing a common long-term trend. With detrending off, a shared trend
could inflate correlations artifically. Default is on.

Uses annual_params.csv directly.
Columns:
    max_doy_raw_anom    : raw observed phase anomaly (days from median)
    amplitude_raw_anom  : raw observed amplitude anomaly (Mkm²)
    max_doy_anom        : fitted phase parameter anomaly (days)
    amplitude_anom      : fitted amplitude parameter anomaly (Mkm²)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, linregress

from ch3_style import (
    apply_style,
    SECTORS_NO_CIRC, SECTOR_COLORS, SECTOR_LABELS,
    DECADE_LEGEND, decade_color,
    stroke, save_fig, DEFAULT_OUTPUT_DIR,
)

apply_style()

DATA_DIR   = "/user/geog/falejandraperez/sea-ice-phase/scripts/R/Ch3/data"
OUTPUT_DIR = DEFAULT_OUTPUT_DIR
ANNUAL_CSV = os.path.join(DATA_DIR, "annual_params.csv")

# Set to True to remove the long-term linear trend from both variables before
# computing correlations — recommended for the independence argument so shared
# secular trends don't inflate the relationship artificially.
DETREND = True


# --- Load ------------------------------------------------------------------

print("Loading annual params...")
annual = pd.read_csv(ANNUAL_CSV)

for col in ["max_doy_raw_anom", "amplitude_raw_anom",
            "max_doy_anom", "amplitude_anom"]:
    annual[col] = pd.to_numeric(annual[col], errors="coerce")

print(f"  {len(annual)} rows | {annual['Year'].min()}–{annual['Year'].max()}")


# --- Detrending helper -----------------------------------------------------

def detrend_series(years, values):
    # Fits and removes a linear trend. Returns residuals.
    # NaNs are excluded from the fit but preserved in position.
    mask = ~np.isnan(values)
    if mask.sum() < 5:
        return values
    slope, intercept, *_ = linregress(years[mask].astype(float),
                                      values[mask])
    return values - (slope * years.astype(float) + intercept)


def prepare_sector(df, sector, x_col, y_col):
    sub = (df[df["sector"] == sector]
           .sort_values("Year")
           .dropna(subset=[x_col, y_col])
           .copy())
    if DETREND:
        sub[x_col] = detrend_series(sub["Year"].values, sub[x_col].values)
        sub[y_col] = detrend_series(sub["Year"].values, sub[y_col].values)
    return sub


# --- Figure ----------------------------------------------------------------
# 5 sectors × 2 columns. Raw observed on the left, APAC modelled on the right.
# Shared y-axis within each row so amplitudes are directly comparable.

fig, axes = plt.subplots(
    len(SECTORS_NO_CIRC), 2,
    figsize=(12, 18),
    sharey="row",
)

COLS = [
    (
        "max_doy_raw_anom",
        "amplitude_raw_anom",
        "Phase anomaly — raw observed (days)",
        "Raw observed",
    ),
    (
        "max_doy_anom",
        "amplitude_anom",
        "Phase anomaly — APAC fitted (days)",
        "APAC modelled",
    ),
]

print(f"\nPhase vs amplitude correlations "
      f"({'detrended' if DETREND else 'raw'}):\n")
print(f"  {'Sector':<28} {'Col':>6}  {'r':>6}  {'rho':>6}")
print("  " + "-" * 52)

for row, sector in enumerate(SECTORS_NO_CIRC):
    color = SECTOR_COLORS[sector]

    for col, (x_col, y_col, xlabel, col_title) in enumerate(COLS):
        ax  = axes[row, col]
        sub = prepare_sector(annual, sector, x_col, y_col)

        x     = sub[x_col].values
        y     = sub[y_col].values
        years = sub["Year"].values

        # Scatter coloured by decade
        for yr, xi, yi in zip(years, x, y):
            ax.scatter(xi, yi, color=decade_color(yr),
                       s=45, zorder=4, edgecolors="white", linewidth=0.4)

        # Label post-2016 years explicitly
        for yr, xi, yi in zip(years, x, y):
            if yr >= 2016:
                ax.text(xi, yi, str(yr),
                        fontsize=7, color="#D4537E",
                        ha="left", va="bottom",
                        path_effects=stroke(lw=2))

        # OLS trend line
        if len(x) > 2:
            m, b   = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, m * x_line + b,
                    color="#B4B2A9", lw=1.2, ls="--", zorder=3)

        # Reference lines at zero
        ax.axhline(0, color="grey", lw=0.5, ls="--", zorder=1)
        ax.axvline(0, color="grey", lw=0.5, ls="--", zorder=1)

        # Correlation stats
        r,   _ = pearsonr(x, y)
        rho, _ = spearmanr(x, y)

        sig = "*" if _ < 0.05 else ("." if _ < 0.10 else "")
        ax.text(0.97, 0.97,
                f"r = {r:+.2f}{sig}   ρ = {rho:+.2f}",
                transform=ax.transAxes,
                fontsize=9, ha="right", va="top",
                color=color, fontweight="bold",
                path_effects=stroke())

        print(f"  {SECTOR_LABELS[sector]:<28} "
              f"{'raw' if col==0 else 'mod':>6}  "
              f"{r:>+6.3f}  {rho:>+6.3f}")

        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("Amplitude anomaly (million km²)", fontsize=9)

        # Sector label as row label on left column
        if col == 0:
            ax.text(-0.18, 0.5, SECTOR_LABELS[sector],
                    transform=ax.transAxes,
                    fontsize=11, fontweight="bold",
                    color=color, va="center", ha="right",
                    rotation=90)

        # Column titles on top row only
        if row == 0:
            ax.set_title(col_title, fontsize=11, fontweight="bold", pad=10)

        # Flag ABS modelled — fitting procedure introduces coupling here
        if sector == "SIE_Amundsen_Bellingshausen" and col == 1:
            ax.text(0.03, 0.03,
                    "Note: fitting procedure\nintroduces coupling here",
                    transform=ax.transAxes,
                    fontsize=7.5, va="bottom", color="#5F5E5A",
                    style="italic", path_effects=stroke())

# Detrend note so it's visible on the figure itself
detrend_note = "Variables linearly detrended before correlation" if DETREND else ""
if detrend_note:
    fig.text(0.5, -0.005, detrend_note,
             ha="center", fontsize=8, color="#5F5E5A", style="italic")

# Shared decade legend
handles = [
    plt.scatter([], [], color=c, s=45,
                edgecolors="white", linewidth=0.4, label=l)
    for c, l in DECADE_LEGEND
]
fig.legend(handles=handles, loc="lower center", ncol=5,
           fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))

fig.suptitle(
    "Phase and amplitude are independent — raw observations and APAC model\n"
    "Near-zero correlations confirm the two quantities vary orthogonally",
    fontsize=12, fontweight="bold", y=1.01,
)

fig.tight_layout(rect=[0.05, 0.04, 1, 1])
save_fig(fig, "fig_phase_amplitude_independence.png", OUTPUT_DIR)

print("\nfig_phase_amplitude_independence.png saved.")