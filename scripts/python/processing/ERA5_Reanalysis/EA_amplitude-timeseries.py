"""
ea_amplitude_sam_timeseries.py

Two-panel figure showing East Antarctica amplitude anomaly and SAM annual
on the same axes, with a vertical line at 2016.

Makes the visual case for the structural break before any statistics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os

# =============================================================================
# PATHS
# =============================================================================

ANNUAL_CSV  = "/user/geog/falejandraperez/sea-ice-phase/scripts/R/chapter3/data/annual_params.csv"
INDEX_CSV   = "/user/geog/falejandraperez/sea-ice-phase/scripts/R/chapter3/figures/master_index_detrended.csv"
OUTPUT_DIR  = "/user/geog/falejandraperez/sea-ice-phase/scripts/R/chapter3/figures/"

YEAR_MIN = 1979
YEAR_MAX = 2023

# =============================================================================
# LOAD AND PREPARE DATA
# =============================================================================

annual = pd.read_csv(ANNUAL_CSV)
idx    = pd.read_csv(INDEX_CSV)

# East Antarctica amplitude anomaly
ea = (annual[(annual["sector"] == "SIE_East_Antarctica") &
             (annual["Year"].between(YEAR_MIN, YEAR_MAX))]
     [["Year", "amplitude_anom"]].dropna().sort_values("Year"))

# Detrend EA amplitude
x = ea["Year"].values.astype(float)
y = ea["amplitude_anom"].values.astype(float)
slope, intercept, _, _, _ = stats.linregress(x, y)
ea["amplitude_dt"] = y - (slope * x + intercept)

# SAM annual — already detrended in master index
sam = idx[idx["Year"].between(YEAR_MIN, YEAR_MAX)][["Year","SAM_annual"]].dropna()

# Merge
merged = ea.merge(sam, on="Year", how="inner").dropna()

# Normalise SAM to same scale as amplitude for visual comparison
# Z-score both series so they plot on the same axis
merged["ea_z"]  = (merged["amplitude_dt"]  - merged["amplitude_dt"].mean())  / merged["amplitude_dt"].std()
merged["sam_z"] = (merged["SAM_annual"] - merged["SAM_annual"].mean()) / merged["SAM_annual"].std()

# Pre and post 2016 correlations
pre  = merged[merged["Year"] <= 2015]
post = merged[merged["Year"] >= 2016]
r_full, _ = stats.pearsonr(merged["SAM_annual"], merged["amplitude_dt"])
r_pre,  _ = stats.pearsonr(pre["SAM_annual"],  pre["amplitude_dt"])
r_post, p_post = stats.pearsonr(post["SAM_annual"], post["amplitude_dt"]) if len(post) > 3 else (np.nan, np.nan)

# =============================================================================
# FIGURE
# =============================================================================

fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
fig.subplots_adjust(hspace=0.08, top=0.92, bottom=0.10, left=0.09, right=0.97)

colors = {"ea": "#185FA5", "sam": "#D85A30"}

# --- Panel 1: Raw time series (z-scored for comparison) ---
ax1 = axes[0]

ax1.axvspan(2016, YEAR_MAX + 1, alpha=0.08, color="#E24B4A", zorder=0)
ax1.axhline(0, color="#888780", linewidth=0.5, alpha=0.4, zorder=1)
ax1.axvline(2016, color="#E24B4A", linewidth=1.0, linestyle="--", alpha=0.6, zorder=2)

ax1.plot(merged["Year"], merged["ea_z"], color=colors["ea"],
         linewidth=2.0, zorder=3, label="EA amplitude anomaly")
ax1.plot(merged["Year"], merged["sam_z"], color=colors["sam"],
         linewidth=1.5, linestyle="--", zorder=3, alpha=0.8, label="SAM annual")

# Shade years where both are same sign (co-varying) vs opposite sign (decoupled)
for _, row in merged.iterrows():
    same_sign = (row["ea_z"] * row["sam_z"]) > 0
    ax1.axvspan(row["Year"] - 0.5, row["Year"] + 0.5,
                alpha=0.06,
                color=colors["ea"] if same_sign else "#888780",
                zorder=0)

ax1.set_ylabel("Standardised anomaly", fontsize=10)
ax1.set_ylim(-3, 3)
ax1.spines[["top","right"]].set_visible(False)
ax1.legend(loc="upper left", fontsize=9, frameon=False)
ax1.text(0.99, 0.97, "Blue shading: co-varying years | Grey: decoupled",
         transform=ax1.transAxes, fontsize=8, va="top", ha="right",
         color="#888780")

# Correlation annotations
ax1.text(2000, 2.5, f"Pre-2016: r = {r_pre:+.2f}",
         fontsize=9, color=colors["ea"], fontweight="bold")
ax1.text(2017.2, 2.5, f"Post-2016: r = {r_post:+.2f}",
         fontsize=9, color="#888780", fontweight="bold")

# --- Panel 2: Scatter pre vs post 2016 ---
ax2 = axes[1]

ax2.axhline(0, color="#888780", linewidth=0.5, alpha=0.4)
ax2.axvline(0, color="#888780", linewidth=0.5, alpha=0.4)

# Pre-2016 scatter
ax2.scatter(pre["SAM_annual"], pre["amplitude_dt"],
            color=colors["ea"], s=40, zorder=3, alpha=0.8,
            label=f"Pre-2016 (r={r_pre:+.2f})")

# Post-2016 scatter
ax2.scatter(post["SAM_annual"], post["amplitude_dt"],
            color="#E24B4A", s=60, zorder=4, alpha=0.9,
            marker="D", label=f"Post-2016 (r={r_post:+.2f})")

# Label post-2016 years
for _, row in post.iterrows():
    ax2.text(row["SAM_annual"] + 0.03, row["amplitude_dt"] + 0.01,
             str(int(row["Year"])), fontsize=7.5, color="#E24B4A", alpha=0.9)

# Pre-2016 regression line
x_line = np.linspace(pre["SAM_annual"].min(), pre["SAM_annual"].max(), 100)
s, i, _, _, _ = stats.linregress(pre["SAM_annual"], pre["amplitude_dt"])
ax2.plot(x_line, s * x_line + i, color=colors["ea"],
         linewidth=1.5, linestyle="--", alpha=0.6, zorder=2)

ax2.set_xlabel("SAM annual (detrended)", fontsize=10)
ax2.set_ylabel("EA amplitude anomaly\n(Mkm², detrended)", fontsize=10)
ax2.legend(loc="upper left", fontsize=9, frameon=False)
ax2.spines[["top","right"]].set_visible(False)
ax2.text(0.99, 0.05,
         "Dashed line: pre-2016 regression\nRed diamonds: post-2016 years",
         transform=ax2.transAxes, fontsize=8, va="bottom", ha="right",
         color="#888780")

# --- Title ---
fig.suptitle(
    "East Antarctica amplitude anomaly vs SAM annual — evidence for structural change\n"
    f"Full record r = {r_full:+.2f} | Pre-2016 r = {r_pre:+.2f} | Post-2016 r = {r_post:+.2f}",
    fontsize=11
)

outpath = os.path.join(OUTPUT_DIR, "ea_amplitude_sam_timeseries.png")
fig.savefig(outpath, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Figure saved: {outpath}")
print(f"\nKey values:")
print(f"  Full record r = {r_full:+.3f}")
print(f"  Pre-2016 r    = {r_pre:+.3f}")
print(f"  Post-2016 r   = {r_post:+.3f} (n={len(post)})")

# =============================================================================
# SYNC TO GOOGLE DRIVE
# =============================================================================
import subprocess

result = subprocess.run([
    "rclone", "copy", outpath,
    "gdrive:sea-ice-phase/chapter3/figures/"
], capture_output=True, text=True)
if result.returncode == 0:
    print("Synced to Google Drive")
else:
    print(f"rclone error: {result.stderr}")
