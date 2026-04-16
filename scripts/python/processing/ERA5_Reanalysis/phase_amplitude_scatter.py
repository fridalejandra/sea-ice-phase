"""
phase_amplitude_scatter.py

Scatter plot of phase anomaly vs amplitude anomaly for each sector.
Tests whether phase and amplitude are empirically independent in the data,
even though they are not constrained to be orthogonal by the APAC model.

If independent: scatter cloud with no trend, r ≈ 0
If correlated: systematic pattern — e.g. early retreat years also
               tend to have smaller amplitude cycles

Points are coloured by decade to show any temporal clustering.
Post-2016 years labelled explicitly.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import stats
import os

# =============================================================================
# PATHS
# =============================================================================

ANNUAL_CSV = "/user/geog/falejandraperez/sea-ice-phase/scripts/R/chapter3/data/annual_params.csv"
OUTPUT_DIR = "/user/geog/falejandraperez/sea-ice-phase/scripts/R/chapter3/figures/"
YEAR_MIN, YEAR_MAX = 1979, 2023

SECTORS = {
    "SIE_Weddell":                 "Weddell",
    "SIE_Amundsen_Bellingshausen": "ABS",
    "SIE_Ross":                    "Ross",
    "SIE_East_Antarctica":         "East Antarctica",
    "SIE_King_Haakon":             "King Haakon",
}

# Decade colour map
def decade_color(year):
    if year < 1990: return "#2C2C2A"    # dark gray — 1980s
    elif year < 2000: return "#185FA5"  # blue — 1990s
    elif year < 2010: return "#1D9E75"  # teal — 2000s
    elif year < 2016: return "#BA7517"  # amber — 2010-2015
    else: return "#E24B4A"              # red — post-2016

# =============================================================================
# LOAD AND DETREND
# =============================================================================

annual = pd.read_csv(ANNUAL_CSV)
annual = annual[annual["Year"].between(YEAR_MIN, YEAR_MAX)]

def detrend(x, y):
    mask = ~np.isnan(y)
    if mask.sum() < 5:
        return y
    s, i, _, _, _ = stats.linregress(x[mask], y[mask])
    return y - (s * x + i)

annual_dt = []
for sec_col in SECTORS.keys():
    sec = annual[annual["sector"] == sec_col].copy().sort_values("Year")
    x = sec["Year"].values.astype(float)
    for var in ["max_doy_anom", "amplitude_anom"]:
        sec[var] = detrend(x, sec[var].values.astype(float))
    annual_dt.append(sec)
annual_dt = pd.concat(annual_dt)

# =============================================================================
# FIGURE — 5 panels, one per sector
# =============================================================================

fig, axes = plt.subplots(2, 3, figsize=(14, 9))
axes = axes.flatten()
fig.subplots_adjust(hspace=0.38, wspace=0.32,
                    top=0.92, bottom=0.10, left=0.07, right=0.97)

print("Phase vs Amplitude correlations by sector:")
print(f"{'Sector':<20} {'Pearson r':>10} {'p':>8} {'Spearman rho':>14} {'p':>8}")
print("-" * 65)

for ax, (sec_col, sec_label) in zip(axes, SECTORS.items()):
    sec = annual_dt[annual_dt["sector"] == sec_col][
        ["Year", "max_doy_anom", "amplitude_anom"]].dropna()

    x = sec["max_doy_anom"].values    # phase anomaly (days)
    y = sec["amplitude_anom"].values  # amplitude anomaly (Mkm²)
    years = sec["Year"].values.astype(int)

    # Correlations
    r, p     = stats.pearsonr(x, y)
    rho, p_s = stats.spearmanr(x, y)
    print(f"  {sec_label:<18} {r:>+10.3f} {p:>8.4f} {rho:>+14.3f} {p_s:>8.4f}")

    # Scatter — colour by decade
    colors = [decade_color(yr) for yr in years]
    ax.scatter(x, y, c=colors, s=35, zorder=3, alpha=0.85)

    # Label post-2016 years
    post = sec[sec["Year"] >= 2016]
    for _, row in post.iterrows():
        ax.annotate(str(int(row["Year"])),
                    (row["max_doy_anom"], row["amplitude_anom"]),
                    fontsize=7, color="#E24B4A",
                    xytext=(4, 3), textcoords="offset points")

    # Regression line
    x_line = np.linspace(x.min(), x.max(), 100)
    s_r, i_r, _, _, _ = stats.linregress(x, y)
    ax.plot(x_line, s_r * x_line + i_r, color="#888780",
            linewidth=1.2, linestyle="--", alpha=0.6, zorder=2)

    # Reference lines
    ax.axhline(0, color="#888780", linewidth=0.5, alpha=0.4)
    ax.axvline(0, color="#888780", linewidth=0.5, alpha=0.4)

    # Formatting
    sig_str = "*" if p < 0.05 else ("." if p < 0.10 else "")
    ax.set_title(f"{sec_label}\nr = {r:+.2f}{sig_str}  ρ = {rho:+.2f}",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Phase anomaly (days)", fontsize=9)
    ax.set_ylabel("Amplitude anomaly (Mkm²)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.spines[["top","right"]].set_visible(False)

# Hide unused sixth panel
axes[5].set_visible(False)

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0],[0], marker="o", color="w", markerfacecolor="#2C2C2A",
           markersize=7, label="1980s"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor="#185FA5",
           markersize=7, label="1990s"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor="#1D9E75",
           markersize=7, label="2000s"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor="#BA7517",
           markersize=7, label="2010–2015"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor="#E24B4A",
           markersize=7, label="Post-2016"),
]
axes[5].set_visible(True)
axes[5].axis("off")
axes[5].legend(handles=legend_elements, loc="center",
               fontsize=9, frameon=False, title="Decade",
               title_fontsize=9)

fig.suptitle("Phase anomaly vs Amplitude anomaly", fontsize=11)

outpath = os.path.join(OUTPUT_DIR, "phase_amplitude_scatter.png")
fig.savefig(outpath, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\nFigure saved: {outpath}")

# Sync to Google Drive
import subprocess
result = subprocess.run([
    "rclone", "copy", outpath,
    "gdrive:results/Ch3_Figures/"
], capture_output=True, text=True)
if result.returncode == 0:
    print("Synced to gdrive:results/Ch3_Figures/")
else:
    print(f"rclone error: {result.stderr}")