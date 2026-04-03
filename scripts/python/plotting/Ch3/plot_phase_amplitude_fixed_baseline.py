"""
plot_phase_amplitude_fixed_baseline.py
Plots phase and amplitude anomalies using 1979-2000 fixed baseline.
Presentation style — coloured by decade, 5-year running mean,
vertical 2016 marker, no std dev band.
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

# Decade colours
def decade_color(year):
    if year < 1990:   return "#888780"   # 1980s — grey
    elif year < 2000: return "#378ADD"   # 1990s — blue
    elif year < 2010: return "#1D9E75"   # 2000s — teal
    elif year < 2016: return "#BA7517"   # 2010-2015 — amber
    else:             return "#D4537E"   # 2016+ — pink-red

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

# Fixed 1979-2000 baseline per sector
for var in ["max_doy", "amplitude"]:
    baseline_col = f"{var}_baseline"
    anom_col     = f"{var}_anom_fixed"
    annual[baseline_col] = annual.groupby("sector")["Year"].transform(
        lambda x: x.between(1979, 2000)
    )
    baselines = (annual[annual["Year"].between(1979, 2000)]
                 .groupby("sector")[var].median())
    annual[anom_col] = (annual[var] -
                        annual["sector"].map(baselines))

print("Fixed baseline anomalies computed")
print(annual.groupby("sector")[["max_doy_anom_fixed","amplitude_anom_fixed"]].mean())

# =============================================================================
# PLOTTING FUNCTION
# =============================================================================

def plot_anomaly_panel(var, ylabel, title, outfile, zero_label=""):

    fig, axes = plt.subplots(1, 5, figsize=(18, 5),
                             sharey=False, sharex=True)

    for ax, (sec_col, sec_label) in zip(axes, SECTORS.items()):
        sub = annual[annual["sector"] == sec_col].sort_values("Year")
        yrs  = sub["Year"].values
        vals = sub[var].values

        # Zero line
        ax.axhline(0, color="#B4B2A9", lw=0.8, ls="--", zorder=1)

        # 2016 vertical marker
        ax.axvline(2016, color="#D4537E", lw=1.2, ls="--",
                   alpha=0.7, zorder=2)

        # Post-2016 mean line
        post_mean = sub[sub["Year"] >= 2016][var].mean()
        ax.axhline(post_mean, color="#D4537E", lw=1.0,
                   ls=":", alpha=0.6, zorder=2)

        # Scatter coloured by decade
        for yr, val in zip(yrs, vals):
            ax.scatter(yr, val, color=decade_color(yr),
                       s=45, zorder=4, edgecolors="white",
                       linewidth=0.5)

        # 5-year running mean
        if len(vals) >= 5:
            smooth = uniform_filter1d(vals, size=5, mode="nearest")
            ax.plot(yrs, smooth, color="#2C2C2A", lw=1.8,
                    zorder=3, alpha=0.8)

        ax.set_title(sec_label, fontsize=12,
                     fontweight="bold", pad=8)
        ax.tick_params(labelsize=10)
        ax.set_xlim(1977, 2024)

        # Y label only on leftmost
        if ax == axes[0]:
            ax.set_ylabel(ylabel, fontsize=11, labelpad=8)

        # X label
        ax.set_xlabel("Year", fontsize=10, color="#5F5E5A")

        # Annotate post-2016 mean
        ax.text(0.97, 0.04,
                f"Post-2016: {post_mean:+.1f}d" if "doy" in var
                else f"Post-2016: {post_mean:+.2f}",
                transform=ax.transAxes, fontsize=8,
                color="#D4537E", ha="right",
                path_effects=[pe.withStroke(linewidth=2,
                              foreground="white")])

    # Shared legend
    handles = [plt.scatter([], [], color=c, s=45,
                           edgecolors="white", linewidth=0.5,
                           label=l)
               for c, l in DECADE_LEGEND]
    handles.append(plt.Line2D([0],[0], color="#2C2C2A",
                               lw=1.8, label="5-yr running mean"))

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
    var      = "max_doy_anom_fixed",
    ylabel   = "Phase anomaly (days)\n← Ahead of phase  |  Behind phase →",
    title    = "Timing of Sea Ice Maximum — Anomaly from 1979–2000 Baseline",
    outfile  = "2_phase_anomaly_fixed_baseline.png"
)

plot_anomaly_panel(
    var      = "amplitude_anom_fixed",
    ylabel   = "Amplitude anomaly (million km²)\n← Smaller  |  Larger →",
    title    = "Seasonal Amplitude Anomaly — Deviation from 1979–2000 Baseline",
    outfile  = "3_amplitude_anomaly_fixed_baseline.png"
)

print("\n=== Done ===")