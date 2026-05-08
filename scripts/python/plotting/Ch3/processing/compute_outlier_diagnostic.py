"""
compute_outlier_diagnostic.py

Identifies outlier years driving key correlations by computing:
1. Leverage (influence on regression line)
2. Studentized residuals (distance from fitted line)
3. Cook's distance (combined influence measure)

Focus: Ross phase anomaly ~ ASL DJF
Also checks: EA amplitude ~ SAM annual for comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os

# =============================================================================
# PATHS
# =============================================================================

ANNUAL_CSV = "/user/geog/falejandraperez/sea-ice-phase/scripts/R/chapter3/data/annual_params.csv"
INDEX_CSV  = "/user/geog/falejandraperez/sea-ice-phase/scripts/R/chapter3/figures/master_index_detrended.csv"
OUTPUT_DIR = "/user/geog/falejandraperez/sea-ice-phase/scripts/R/chapter3/figures/"
YEAR_MIN, YEAR_MAX = 1979, 2023

# =============================================================================
# LOAD AND DETREND
# =============================================================================

annual = pd.read_csv(ANNUAL_CSV)
annual = annual[annual["Year"].between(YEAR_MIN, YEAR_MAX)]
idx    = pd.read_csv(INDEX_CSV)

def detrend(df, year_col, val_col):
    df = df.copy().dropna(subset=[val_col]).sort_values(year_col)
    x = df[year_col].values.astype(float)
    y = df[val_col].values.astype(float)
    s, i, _, _, _ = stats.linregress(x, y)
    df[val_col] = y - (s * x + i)
    return df

# Ross phase
ross = (annual[annual["sector"] == "SIE_Ross"]
        [["Year","max_doy_anom"]].dropna().sort_values("Year"))
ross = detrend(ross, "Year", "max_doy_anom")

# EA amplitude
ea = (annual[annual["sector"] == "SIE_East_Antarctica"]
      [["Year","amplitude_anom"]].dropna().sort_values("Year"))
ea = detrend(ea, "Year", "amplitude_anom")

# =============================================================================
# OUTLIER DIAGNOSTIC FUNCTION
# =============================================================================

def outlier_report(x_vals, y_vals, years, x_label, y_label, title):
    """
    Compute leverage, studentized residuals, and Cook's distance.
    Print ranked table and return figure.
    """
    n = len(x_vals)
    # Fit OLS
    slope, intercept, r, p, _ = stats.linregress(x_vals, y_vals)
    y_hat    = slope * x_vals + intercept
    residuals = y_vals - y_hat
    sse      = np.sum(residuals**2)
    mse      = sse / (n - 2)

    # Leverage (hat values) for simple regression
    x_mean = x_vals.mean()
    sxx    = np.sum((x_vals - x_mean)**2)
    h      = 1/n + (x_vals - x_mean)**2 / sxx

    # Studentized residuals
    stud_resid = residuals / (np.sqrt(mse * (1 - h)) + 1e-12)

    # Cook's distance
    cooks_d = (stud_resid**2 * h) / (2 * (1 - h + 1e-12))

    # Build results table
    df = pd.DataFrame({
        "Year":       years,
        "x":          x_vals.round(3),
        "y":          y_vals.round(3),
        "residual":   residuals.round(3),
        "leverage":   h.round(3),
        "stud_resid": stud_resid.round(2),
        "cooks_d":    cooks_d.round(4),
    }).sort_values("cooks_d", ascending=False)

    print(f"\n{'='*65}")
    print(f"{title}")
    print(f"Full-record r = {r:+.3f}, p = {p:.4f}, n = {n}")
    print(f"{'='*65}")
    print(f"{'Year':>6} {'x':>8} {'y':>8} {'Resid':>8} "
          f"{'Leverage':>9} {'StudRes':>8} {'Cooks D':>9}")
    print("-" * 65)
    for _, row in df.head(10).iterrows():
        flag = " <<<" if abs(row["stud_resid"]) > 2 else ""
        print(f"  {int(row['Year']):4d}  {row['x']:8.3f}  {row['y']:8.3f}  "
              f"{row['residual']:8.3f}  {row['leverage']:9.3f}  "
              f"{row['stud_resid']:8.2f}  {row['cooks_d']:9.4f}{flag}")

    # Leave-one-out correlation — how much does each year change r?
    print(f"\n  Leave-one-out: years that change |r| most:")
    print(f"  {'Year':>6}  {'r without':>10}  {'delta r':>10}")
    loo = []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        r_loo, _ = stats.pearsonr(x_vals[mask], y_vals[mask])
        loo.append((years[i], r_loo, r_loo - r))
    loo.sort(key=lambda x: abs(x[2]), reverse=True)
    for yr, r_loo, delta in loo[:8]:
        print(f"  {int(yr):6d}  {r_loo:+10.3f}  {delta:+10.3f}")

    return df

# =============================================================================
# RUN DIAGNOSTICS
# =============================================================================

# 1. Ross phase ~ ASL DJF
asl = idx[idx["Year"].between(YEAR_MIN, YEAR_MAX)][["Year","ASL_DJF"]].dropna()
merged_ross = ross.merge(asl, on="Year").dropna()
x = merged_ross["ASL_DJF"].values
y = merged_ross["max_doy_anom"].values
years = merged_ross["Year"].values

ross_df = outlier_report(x, y, years,
    "ASL DJF", "Ross phase anomaly (days)",
    "Ross phase anomaly ~ ASL DJF")

# 2. EA amplitude ~ SAM annual (for comparison — known robust result)
sam = idx[idx["Year"].between(YEAR_MIN, YEAR_MAX)][["Year","SAM_annual"]].dropna()
merged_ea = ea.merge(sam, on="Year").dropna()
x2 = merged_ea["SAM_annual"].values
y2 = merged_ea["amplitude_anom"].values
years2 = merged_ea["Year"].values

ea_df = outlier_report(x2, y2, years2,
    "SAM annual", "EA amplitude anomaly (Mkm²)",
    "EA amplitude anomaly ~ SAM annual (reference)")

# =============================================================================
# FIGURE — side by side scatter with Cook's D bubble size
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.subplots_adjust(wspace=0.30, top=0.90, bottom=0.12)

for ax, merged, x_col, y_col, y_label, title, color in [
    (axes[0], merged_ross, "ASL_DJF", "max_doy_anom",
     "Ross phase anomaly (days)", "Ross phase ~ ASL DJF", "#1D9E75"),
    (axes[1], merged_ea, "SAM_annual", "amplitude_anom",
     "EA amplitude anomaly (Mkm²)", "EA amplitude ~ SAM annual", "#185FA5"),
]:
    x_vals = merged[x_col].values
    y_vals = merged[y_col].values
    yrs    = merged["Year"].values.astype(int)
    r, p   = stats.pearsonr(x_vals, y_vals)
    rho, _ = stats.spearmanr(x_vals, y_vals)

    # Size points by Cook's distance for visual outlier detection
    slope, intercept, _, _, _ = stats.linregress(x_vals, y_vals)
    y_hat = slope * x_vals + intercept
    resid = y_vals - y_hat
    mse   = np.sum(resid**2) / (len(x_vals) - 2)
    x_mean = x_vals.mean()
    sxx    = np.sum((x_vals - x_mean)**2)
    h      = 1/len(x_vals) + (x_vals - x_mean)**2 / sxx
    stud   = resid / (np.sqrt(mse * (1 - h)) + 1e-12)
    cooks  = (stud**2 * h) / (2 * (1 - h + 1e-12))
    sizes  = 30 + 300 * (cooks / cooks.max())

    # Colour post-2016 red
    colors = ["#E24B4A" if yr >= 2016 else color for yr in yrs]

    ax.scatter(x_vals, y_vals, s=sizes, c=colors, alpha=0.8, zorder=3)

    # Label high Cook's D years and post-2016
    for i, yr in enumerate(yrs):
        if cooks[i] > 0.1 or yr >= 2016:
            ax.annotate(str(yr), (x_vals[i], y_vals[i]),
                        fontsize=7.5, xytext=(4, 3),
                        textcoords="offset points",
                        color="#E24B4A" if yr >= 2016 else "#2C2C2A")

    # Regression line
    x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
    ax.plot(x_line, slope * x_line + intercept,
            color=color, linewidth=1.5, linestyle="--", alpha=0.6)

    ax.axhline(0, color="#888780", linewidth=0.5, alpha=0.4)
    ax.axvline(0, color="#888780", linewidth=0.5, alpha=0.4)
    ax.set_xlabel(x_col.replace("_", " "), fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.set_title(f"{title}\nr = {r:+.3f}  ρ = {rho:+.3f}  p = {p:.4f}",
                 fontsize=10, fontweight="bold")
    ax.text(0.98, 0.04, "Red = post-2016",
            transform=ax.transAxes, fontsize=7.5, ha="right",
            color="#888780")
    ax.spines[["top","right"]].set_visible(False)

fig.suptitle("", fontsize=11)

outpath = os.path.join(OUTPUT_DIR, "outlier_diagnostic.png")
fig.savefig(outpath, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\nFigure saved: {outpath}")

# Sync to Google Drive
import subprocess
result = subprocess.run(
    ["rclone", "copy", outpath, "gdrive:results/Ch3_Figures/"],
    capture_output=True, text=True
)
if result.returncode == 0:
    print("Synced to gdrive:results/Ch3_Figures/")
else:
    print(f"rclone error: {result.stderr}")