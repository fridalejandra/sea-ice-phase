"""
rolling_correlations.py
Computes 15-year rolling Pearson r between APAC phase/amplitude anomalies
and key atmospheric indices. Plots temporal evolution with post-2016 shading.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy import stats
import os

# =============================================================================
# 0. PATHS
# =============================================================================

ANNUAL_CSV  = "/user/geog/falejandraperez/sea-ice-phase/scripts/R/chapter3/data/annual_params.csv"
INDICES_DIR = "/user/geog/falejandraperez/sea-ice-phase/data/indices/"
OUTPUT_DIR  = "/user/geog/falejandraperez/sea-ice-phase/scripts/R/chapter3/figures/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

WINDOW = 15  # years

# =============================================================================
# 1. LOAD AND DETREND APAC
# =============================================================================

annual = pd.read_csv(ANNUAL_CSV)
annual = annual[annual["Year"].between(1979, 2022)]

def detrend_series(df, year_col, val_col):
    df = df.copy().dropna(subset=[val_col]).sort_values(year_col)
    if len(df) < 5:
        return df
    x = df[year_col].values.astype(float)
    y = df[val_col].values.astype(float)
    slope, intercept, _, _, _ = stats.linregress(x, y)
    df[val_col] = y - (slope * x + intercept)
    return df

annual_dt = []
for sec in annual["sector"].unique():
    sec_df = annual[annual["sector"] == sec].copy()
    for var in ["max_doy_anom", "amplitude_anom"]:
        sec_df = detrend_series(sec_df, "Year", var)
    annual_dt.append(sec_df)
annual_dt = pd.concat(annual_dt)

# =============================================================================
# 2. LOAD AND DETREND INDICES
# =============================================================================

month_map = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,
             "Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}

def load_wide(filepath, val_col):
    df = pd.read_csv(filepath, sep=r"\s+", header=0,
                     names=["Year","Jan","Feb","Mar","Apr","May","Jun",
                            "Jul","Aug","Sep","Oct","Nov","Dec"])
    df = df[df["Year"].between(1979, 2022)]
    long = df.melt(id_vars="Year", var_name="mn", value_name=val_col)
    long["month"] = long["mn"].map(month_map)
    return long[["Year","month",val_col]].sort_values(["Year","month"])

def seasonal_mean(df, val_col, months, new_col):
    df = df.copy()
    if set(months) == {12,1,2}:
        df.loc[df["month"]==12, "Year"] += 1
    sub = df[df["month"].isin(months)].groupby("Year")[val_col].mean().reset_index()
    sub.columns = ["Year", new_col]
    return sub

# SAM
sam_long = load_wide(
    os.path.join(INDICES_DIR, "marshall_sam_monthly.txt"), "SAM")
sam_son = seasonal_mean(sam_long, "SAM", [9,10,11], "SAM_SON")

# AAO
aao_long = load_wide(
    os.path.join(INDICES_DIR, "daily_aao_sam.txt"), "AAO")
aao_jja = seasonal_mean(aao_long, "AAO", [6,7,8], "AAO_JJA")

# ZW3 Goyal
zw3g = pd.read_csv(os.path.join(INDICES_DIR, "ZW3_goyal_annual.csv"))
zw3g = zw3g[zw3g["year"].between(1979, 2022)]
zw3g.columns = ["Year","ZW3G_PC1","ZW3G_PC2","ZW3G_magnitude","ZW3G_phase"]

# Merge into one index table
idx = sam_son.merge(aao_jja, on="Year", how="outer")
idx = idx.merge(zw3g[["Year","ZW3G_PC2"]], on="Year", how="outer")

# Detrend
for col in ["SAM_SON","AAO_JJA","ZW3G_PC2"]:
    tmp = detrend_series(idx[["Year",col]].dropna(), "Year", col)
    idx.loc[tmp.index, col] = tmp[col].values

# =============================================================================
# 3. ROLLING CORRELATION FUNCTION
# =============================================================================

def rolling_corr(apac_df, sector, apac_var, idx_df, idx_col, window=15):
    """
    Compute rolling Pearson r centred on each year.
    Returns DataFrame with Year, r, p, n.
    """
    sec = apac_df[apac_df["sector"] == sector][["Year", apac_var]].dropna()
    merged = sec.merge(idx_df[["Year", idx_col]].dropna(), on="Year")
    merged = merged.sort_values("Year").reset_index(drop=True)

    years = merged["Year"].values
    results = []

    for i, yr in enumerate(years):
        half = window // 2
        mask = (years >= yr - half) & (years <= yr + half)
        sub = merged[mask]
        if len(sub) < 8:
            continue
        r, p = stats.pearsonr(sub[idx_col], sub[apac_var])
        results.append({"Year": yr, "r": r, "p": p, "n": len(sub)})

    return pd.DataFrame(results)

# =============================================================================
# 4. COMPUTE ROLLING CORRELATIONS FOR KEY PAIRS
# =============================================================================

pairs = [
    # (sector_col, apac_var, idx_col, label, color)
    ("SIE_Weddell",                "max_doy_anom",   "ZW3G_PC2",
     "Weddell phase vs ZW3 (Goyal PC2)",    "#7F77DD"),
    ("SIE_Amundsen_Bellingshausen","amplitude_anom", "AAO_JJA",
     "ABS amplitude vs AAO (JJA)",           "#D4537E"),
    ("SIE_East_Antarctica",        "max_doy_anom",   "SAM_SON",
     "East Antarctica phase vs SAM (SON)",   "#1D9E75"),
    ("SIE_Ross",                   "max_doy_anom",   "SAM_SON",
     "Ross phase vs SAM (SON) — contrast",   "#BA7517"),
]

roll_results = {}
for sec_col, apac_var, idx_col, label, color in pairs:
    df = rolling_corr(annual_dt, sec_col, apac_var, idx, idx_col, WINDOW)
    roll_results[label] = (df, color)
    print(f"{label}: {len(df)} rolling windows")

# =============================================================================
# 5. PLOT
# =============================================================================

plt.rcParams.update({"font.family": "Nimbus Sans"})

fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=True, sharex=True)
axes = axes.flatten()

# Significance threshold for n=15, p=0.05 (two-tailed)
r_crit = 0.514

for ax, (label, (df, color)) in zip(axes, roll_results.items()):
    # Significance band
    ax.axhspan(-r_crit, r_crit, color="#F1EFE8", alpha=0.6, zorder=1)
    ax.axhline(0,      color="#B4B2A9", lw=0.8, zorder=2)
    ax.axhline( r_crit, color="#B4B2A9", lw=0.8, ls="--", zorder=2)
    ax.axhline(-r_crit, color="#B4B2A9", lw=0.8, ls="--", zorder=2)

    # Post-2016 shading
    ax.axvspan(2016, 2022, color="#D3D1C7", alpha=0.4, zorder=1)

    # Significant points
    sig  = df[df["p"] < 0.05]
    nsig = df[df["p"] >= 0.05]

    ax.plot(df["Year"], df["r"], color=color, lw=2, zorder=4)
    ax.scatter(sig["Year"],  sig["r"],  color=color, s=40,
               zorder=5, edgecolors="white", linewidth=0.8)
    ax.scatter(nsig["Year"], nsig["r"], color=color, s=20,
               zorder=5, facecolors="none", edgecolors=color,
               linewidth=0.8, alpha=0.6)

    ax.set_title(label, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlim(1979 + WINDOW//2, 2022 - WINDOW//2 + 1)
    ax.set_ylim(-0.85, 0.85)
    ax.tick_params(labelsize=10)

    # Add n label
    ax.text(0.02, 0.04, f"Window = {WINDOW} yr",
            transform=ax.transAxes, fontsize=9,
            color="#5F5E5A",
            path_effects=[pe.withStroke(linewidth=2, foreground="white")])

# Shared labels
fig.text(0.5, 0.02, "Year (window centred)",
         ha="center", fontsize=12)
fig.text(0.02, 0.5, "Pearson r (rolling)",
         va="center", rotation="vertical", fontsize=12)

fig.suptitle(
    f"Temporal stability of atmosphere–sea ice relationships\n"
    f"{WINDOW}-year rolling correlations, linearly detrended  |  "
    f"dashed lines = p<0.05 threshold  |  shading = post-2016",
    fontsize=11, y=1.01
)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "rolling_correlations.png"),
            dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Rolling correlations figure saved")