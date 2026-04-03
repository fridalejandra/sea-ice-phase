"""
Plot_ch3_figures.py
Chapter 3 figures — phase and amplitude anomalies.
Fixed 1979-2000 baseline, decade colours, 5-year running mean.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy.ndimage import uniform_filter1d
import os

# =============================================================================
# SETTINGS
# =============================================================================

ANNUAL_CSV = "/user/geog/falejandraperez/sea-ice-phase/scripts/R/chapter3/data/annual_params.csv"
OUTPUT_DIR = "/user/geog/falejandraperez/sea-ice-phase/scripts/R/chapter3/figures/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family":       "Nimbus Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    1.0,
    "savefig.facecolor": "white",
    "figure.dpi":        150,
    "savefig.dpi":       200,
})

SECTORS = {
    "SIE_Weddell":                 "Weddell",
    "SIE_Amundsen_Bellingshausen": "ABS",
    "SIE_Ross":                    "Ross",
    "SIE_East_Antarctica":         "East Antarctica",
    "SIE_King_Haakon":             "King Haakon",
}

def decade_color(year):
    if year < 1990:   return "#888780"
    elif year < 2000: return "#378ADD"
    elif year < 2010: return "#1D9E75"
    elif year < 2016: return "#BA7517"
    else:             return "#D4537E"

DECADE_LEGEND = [
    ("#888780", "1980s"),
    ("#378ADD", "1990s"),
    ("#1D9E75", "2000s"),
    ("#BA7517", "2010–2015"),
    ("#D4537E", "2016+"),
]

# =============================================================================
# LOAD AND COMPUTE FIXED BASELINE ANOMALIES
# =============================================================================

annual = pd.read_csv(ANNUAL_CSV)
annual = annual[annual["Year"].between(1979, 2022)]

baselines_doy = (annual[annual["Year"].between(1979, 2000)]
                 .groupby("sector")["max_doy"].median())
baselines_amp = (annual[annual["Year"].between(1979, 2000)]
                 .groupby("sector")["amplitude"].median())

annual["max_doy_anom_fixed"]   = (annual["max_doy"] -
                                   annual["sector"].map(baselines_doy))
annual["amplitude_anom_fixed"] = (annual["amplitude"] -
                                   annual["sector"].map(baselines_amp))

print("Fixed baseline anomalies computed")

# =============================================================================
# PLOTTING FUNCTION
# =============================================================================

def plot_anomaly_panel(var, ylabel, title, outfile):

    fig, axes = plt.subplots(1, 5, figsize=(18, 5),
                             sharey=False, sharex=True)

    for ax, (sec_col, sec_label) in zip(axes, SECTORS.items()):
        sub  = annual[annual["sector"] == sec_col].sort_values("Year")
        yrs  = sub["Year"].values
        vals = sub[var].values

        # Zero line
        ax.axhline(0, color="#B4B2A9", lw=0.8, ls="--", zorder=1)

        # 2016 vertical marker
        ax.axvline(2016, color="#D4537E", lw=1.2, ls="--",
                   alpha=0.7, zorder=2)

        # Scatter coloured by decade
        for yr, val in zip(yrs, vals):
            ax.scatter(yr, val, color=decade_color(yr),
                       s=50, zorder=4, edgecolors="white",
                       linewidth=0.5)

        # 5-year running mean
        if len(vals) >= 5:
            smooth = uniform_filter1d(vals, size=5, mode="nearest")
            ax.plot(yrs, smooth, color="#2C2C2A", lw=2.0,
                    zorder=3, alpha=0.85)

        # Post-2016 mean annotation only
        post_mean = sub[sub["Year"] >= 2016][var].mean()
        ax.text(0.97, 0.04,
                f"Post-2016 mean: {post_mean:+.1f} days" if "doy" in var
                else f"Post-2016 mean: {post_mean:+.3f} Mkm²",
                transform=ax.transAxes, fontsize=8,
                color="#D4537E", ha="right",
                path_effects=[pe.withStroke(linewidth=2,
                              foreground="white")])

        ax.set_title(sec_label, fontsize=12,
                     fontweight="bold", pad=8)
        ax.tick_params(labelsize=10)
        ax.set_xlim(1977, 2024)

        if ax == axes[0]:
            ax.set_ylabel(ylabel, fontsize=11, labelpad=8)

        ax.set_xlabel("Year", fontsize=10, color="#5F5E5A")

    # Legend
    handles = [plt.scatter([], [], color=c, s=50,
                           edgecolors="white", linewidth=0.5,
                           label=l)
               for c, l in DECADE_LEGEND]
    handles.append(plt.Line2D([0], [0], color="#2C2C2A",
                               lw=2.0, label="5-yr running mean"))

    fig.legend(handles=handles, loc="lower center",
               ncol=6, fontsize=10,
               bbox_to_anchor=(0.5, -0.05),
               frameon=False)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, outfile),
                dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {outfile}")

# =============================================================================
# GENERATE FIGURES
# =============================================================================

plot_anomaly_panel(
    var     = "max_doy_anom_fixed",
    ylabel  = "Phase anomaly (days)\n← Ahead of phase  |  Behind phase →",
    title   = "Timing of Sea Ice Maximum — Anomaly from 1979–2000 Baseline",
    outfile = "2_phase_anomaly_timeseries.png"
)

plot_anomaly_panel(
    var     = "amplitude_anom_fixed",
    ylabel  = "Amplitude anomaly (million km²)\n← Smaller  |  Larger →",
    title   = "Seasonal Amplitude — Anomaly from 1979–2000 Baseline",
    outfile = "3_amplitude_anomaly_timeseries.png"
)

print("\n=== Done ===")