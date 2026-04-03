"""
compute_atmospheric_correlations.py

Correlates APAC phase and amplitude anomalies with atmospheric indices:
  - Marshall SAM (monthly)
  - AAO (monthly)
  - TPI filtered (monthly)
  - ZW3 Raphael (annual)
  - ZW3 Goyal (annual: PC1, PC2, magnitude, phase)

Both APAC variables and atmospheric indices are linearly detrended
before correlation to remove spurious trend-driven signals.

Outputs:
  - correlations_all.csv       — Pearson r and p-value, all indices
  - correlation_heatmap_annual.png — summary heatmap

Reference period: 1979-2022
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

INDICES_DIR = "/user/geog/falejandraperez/sea-ice-phase/data/indices/"
ANNUAL_CSV  = "/user/geog/falejandraperez/sea-ice-phase/scripts/R/chapter3/data/annual_params.csv"
OUTPUT_DIR  = "/user/geog/falejandraperez/sea-ice-phase/scripts/R/chapter3/figures/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SECTORS = {
    "SIE_Weddell":                 "Weddell",
    "SIE_Amundsen_Bellingshausen": "ABS",
    "SIE_Ross":                    "Ross",
    "SIE_East_Antarctica":         "East Antarctica",
    "SIE_King_Haakon":             "King Haakon",
}

# =============================================================================
# 1. LOAD APAC ANNUAL PARAMETERS
# =============================================================================

annual = pd.read_csv(ANNUAL_CSV)
annual = annual[annual["Year"].between(1979, 2022)]

# =============================================================================
# 2. LOAD ATMOSPHERIC INDICES
# =============================================================================

month_map = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,
             "Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}

def load_wide_index(filepath, val_col):
    """Load a year x month wide-format index file."""
    df = pd.read_csv(
        filepath, sep=r"\s+", header=0,
        names=["Year","Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]
    )
    df = df[df["Year"].between(1979, 2022)]
    long = df.melt(id_vars="Year", var_name="month_name", value_name=val_col)
    long["month"] = long["month_name"].map(month_map)
    return long[["Year","month",val_col]].sort_values(["Year","month"])


def seasonal_mean(df, val_col, season_months, new_col):
    """Compute seasonal mean, handling DJF year alignment."""
    df = df.copy()
    # For DJF shift December to following year
    if set(season_months) == {12, 1, 2}:
        df.loc[df["month"] == 12, "Year"] = df.loc[df["month"] == 12, "Year"] + 1
    sub = df[df["month"].isin(season_months)].groupby("Year")[val_col].mean().reset_index()
    sub.columns = ["Year", new_col]
    return sub


# --- Marshall SAM ---
sam_long = load_wide_index(
    os.path.join(INDICES_DIR, "marshall_sam_monthly.txt"), "SAM")
sam_annual = sam_long.groupby("Year")["SAM"].mean().reset_index()
sam_annual.columns = ["Year","SAM_annual"]
sam_djf = seasonal_mean(sam_long, "SAM", [12,1,2], "SAM_DJF")
sam_mam = seasonal_mean(sam_long, "SAM", [3,4,5],  "SAM_MAM")
sam_jja = seasonal_mean(sam_long, "SAM", [6,7,8],  "SAM_JJA")
sam_son = seasonal_mean(sam_long, "SAM", [9,10,11],"SAM_SON")

# --- AAO ---
aao_long = load_wide_index(
    os.path.join(INDICES_DIR, "daily_aao_sam.txt"), "AAO")
aao_annual = aao_long.groupby("Year")["AAO"].mean().reset_index()
aao_annual.columns = ["Year","AAO_annual"]
aao_djf = seasonal_mean(aao_long, "AAO", [12,1,2], "AAO_DJF")
aao_mam = seasonal_mean(aao_long, "AAO", [3,4,5],  "AAO_MAM")
aao_jja = seasonal_mean(aao_long, "AAO", [6,7,8],  "AAO_JJA")
aao_son = seasonal_mean(aao_long, "AAO", [9,10,11],"AAO_SON")

# --- TPI filtered ---
tpi_rows = []
with open(os.path.join(INDICES_DIR, "tpi_filtered.txt")) as f:
    lines = f.readlines()
for line in lines[1:]:
    parts = line.split()
    if len(parts) != 13:
        continue
    # Skip non-numeric header lines
    try:
        year = int(parts[0])
    except ValueError:
        continue
    # Skip missing value rows
    vals = [float(v) for v in parts[1:]]
    tpi_rows.append([year] + vals)

tpi_raw = pd.DataFrame(tpi_rows,
    columns=["Year","Jan","Feb","Mar","Apr","May","Jun",
             "Jul","Aug","Sep","Oct","Nov","Dec"])
tpi_raw = tpi_raw.replace(-99.0, np.nan)
tpi_raw = tpi_raw[tpi_raw["Year"].between(1979, 2022)]
tpi_long = tpi_raw.melt(id_vars="Year", var_name="month_name", value_name="TPI")
tpi_long["month"] = tpi_long["month_name"].map(month_map)
tpi_long = tpi_long[["Year","month","TPI"]].sort_values(["Year","month"])

tpi_annual = tpi_long.groupby("Year")["TPI"].mean().reset_index()
tpi_annual.columns = ["Year","TPI_annual"]
tpi_djf = seasonal_mean(tpi_long, "TPI", [12,1,2], "TPI_DJF")
tpi_mam = seasonal_mean(tpi_long, "TPI", [3,4,5],  "TPI_MAM")
tpi_jja = seasonal_mean(tpi_long, "TPI", [6,7,8],  "TPI_JJA")
tpi_son = seasonal_mean(tpi_long, "TPI", [9,10,11],"TPI_SON")

# --- ZW3 Raphael ---
zw3r = pd.read_csv(os.path.join(INDICES_DIR, "ZW3_raphael_annual.csv"))
zw3r = zw3r[zw3r["year"].between(1979, 2022)]
zw3r.columns = ["Year","ZW3R_annual"]

# --- ZW3 Goyal ---
zw3g = pd.read_csv(os.path.join(INDICES_DIR, "ZW3_goyal_annual.csv"))
zw3g = zw3g[zw3g["year"].between(1979, 2022)]
zw3g.columns = ["Year","ZW3G_PC1","ZW3G_PC2","ZW3G_magnitude","ZW3G_phase"]

# =============================================================================
# 3. BUILD MASTER INDEX TABLE
# =============================================================================

idx = sam_annual.copy()
for df in [aao_annual, tpi_annual, zw3r, zw3g,
           sam_djf, sam_mam, sam_jja, sam_son,
           aao_djf, aao_mam, aao_jja, aao_son,
           tpi_djf, tpi_mam, tpi_jja, tpi_son]:
    idx = idx.merge(df, on="Year", how="left")

print(f"Index table: {idx.shape}")
print(idx.head())

# =============================================================================
# 4. DETREND ALL SERIES
# =============================================================================

def detrend_series(df, year_col, val_col):
    """Remove linear trend, return df with val_col as residuals."""
    df = df.copy().dropna(subset=[val_col]).sort_values(year_col)
    if len(df) < 5:
        return df
    x = df[year_col].values.astype(float)
    y = df[val_col].values.astype(float)
    slope, intercept, _, _, _ = stats.linregress(x, y)
    df[val_col] = y - (slope * x + intercept)
    return df


# Detrend APAC variables per sector
annual_dt = []
for sec_col in SECTORS.keys():
    sec = annual[annual["sector"] == sec_col].copy()
    for var in ["max_doy_anom", "amplitude_anom"]:
        sec = detrend_series(sec, "Year", var)
    annual_dt.append(sec)
annual_dt = pd.concat(annual_dt)

# Detrend all index columns
INDEX_COLS_ALL = [
    "SAM_annual","AAO_annual","TPI_annual",
    "ZW3R_annual","ZW3G_PC1","ZW3G_PC2","ZW3G_magnitude","ZW3G_phase",
    "SAM_DJF","SAM_MAM","SAM_JJA","SAM_SON",
    "AAO_DJF","AAO_MAM","AAO_JJA","AAO_SON",
    "TPI_DJF","TPI_MAM","TPI_JJA","TPI_SON",
]

idx_dt = idx.copy()
for col in INDEX_COLS_ALL:
    if col in idx_dt.columns:
        tmp = detrend_series(idx_dt[["Year", col]].dropna(), "Year", col)
        idx_dt.loc[tmp.index, col] = tmp[col].values

print("\nDetrending complete — SAM_annual std check:")
print(f"  Before: {idx['SAM_annual'].std():.3f}")
print(f"  After:  {idx_dt['SAM_annual'].std():.3f}")

# =============================================================================
# 5. CORRELATIONS
# =============================================================================

APAC_VARS   = ["max_doy_anom", "amplitude_anom"]
APAC_LABELS = {
    "max_doy_anom":   "Phase anomaly",
    "amplitude_anom": "Amplitude anomaly"
}

results = []

for sec_col, sec_label in SECTORS.items():
    sec_data = annual_dt[annual_dt["sector"] == sec_col].copy()

    for apac_var in APAC_VARS:
        for idx_col in INDEX_COLS_ALL:
            merged = sec_data[["Year", apac_var]].merge(
                idx_dt[["Year", idx_col]], on="Year", how="inner"
            ).dropna()

            if len(merged) < 10:
                continue

            r, p = stats.pearsonr(merged[idx_col], merged[apac_var])
            results.append({
                "sector":   sec_label,
                "apac_var": APAC_LABELS[apac_var],
                "index":    idx_col,
                "r":        round(r, 3),
                "p":        round(p, 4),
                "n":        len(merged),
                "sig":      "*" if p < 0.05 else ("." if p < 0.10 else ""),
            })

corr_df = pd.DataFrame(results)
corr_df.to_csv(os.path.join(OUTPUT_DIR, "correlations_all.csv"), index=False)
print(f"\nCorrelations computed: {len(corr_df)} rows")

## =============================================================================
# 6. HEATMAP — updated: ZW3G_PC2 added, TPI removed
# =============================================================================

plt.rcParams.update({"font.family": "Nimbus Sans"})

HEATMAP_INDICES = [
    "SAM_annual", "SAM_SON", "SAM_DJF",
    "AAO_annual", "AAO_SON", "AAO_DJF",
    "ZW3R_annual", "ZW3G_PC2", "ZW3G_magnitude"
]
HEATMAP_LABELS = [
    "SAM\nannual", "SAM\nSON", "SAM\nDJF",
    "AAO\nannual", "AAO\nSON", "AAO\nDJF",
    "ZW3\nRaphael", "ZW3\nGoyal PC2", "ZW3\nGoyal mag"
]

sector_order = ["Weddell", "ABS", "Ross", "East Antarctica", "King Haakon"]

fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)

for ax, apac_var, title in zip(
    axes,
    ["Phase anomaly", "Amplitude anomaly"],
    ["Phase anomaly (max DOY) vs atmospheric indices",
     "Amplitude anomaly vs atmospheric indices"]
):
    sub = corr_df[
        (corr_df["apac_var"] == apac_var) &
        (corr_df["index"].isin(HEATMAP_INDICES))
    ].pivot(index="sector", columns="index", values="r")
    sub = sub.reindex(sector_order)[HEATMAP_INDICES]
    sub.columns = HEATMAP_LABELS

    sig_sub = corr_df[
        (corr_df["apac_var"] == apac_var) &
        (corr_df["index"].isin(HEATMAP_INDICES))
    ].pivot(index="sector", columns="index", values="sig")
    sig_sub = sig_sub.reindex(sector_order)[HEATMAP_INDICES]
    sig_sub.columns = HEATMAP_LABELS

    im = ax.imshow(sub.values.astype(float),
                   cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")

    ax.set_xticks(range(len(HEATMAP_LABELS)))
    ax.set_xticklabels(HEATMAP_LABELS, fontsize=10)
    ax.set_yticks(range(len(sector_order)))
    ax.set_yticklabels(sector_order, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

    for i in range(len(sector_order)):
        for j in range(len(HEATMAP_LABELS)):
            r_val = sub.values[i, j]
            sig   = sig_sub.values[i, j]
            if not np.isnan(float(r_val)):
                ax.text(j, i, f"{float(r_val):.2f}{sig}",
                        ha="center", va="center",
                        fontsize=9, fontweight="bold",
                        color="white" if abs(float(r_val)) > 0.35
                              else "#2C2C2A",
                        path_effects=[pe.withStroke(
                            linewidth=2,
                            foreground="black" if abs(float(r_val)) > 0.35
                                       else "white")])

# Fixed colorbar — horizontal, below figure
cbar = fig.colorbar(im, ax=axes, label="Pearson r",
                    orientation="horizontal",
                    shrink=0.4, pad=0.20, aspect=30)
cbar.ax.tick_params(labelsize=10)

fig.suptitle(
    "APAC anomalies vs atmospheric indices — 1979–2022\n"
    "* p<0.05   . p<0.10",
    fontsize=11, y=1.02
)
fig.subplots_adjust(bottom=0.25, top=0.88, left=0.05, right=0.95, wspace=0.05)
fig.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap_annual.png"),
            dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Heatmap saved")
# =============================================================================
# 7. TERMINAL PREVIEW
# =============================================================================

print("\nTop 10 significant correlations — phase anomaly:")
phase_sig = corr_df[
    (corr_df["apac_var"] == "Phase anomaly") &
    (corr_df["sig"] == "*")
].sort_values("r", key=abs, ascending=False)
print(phase_sig[["sector","index","r","p","n"]].head(10).to_string(index=False))

print("\nTop 10 significant correlations — amplitude anomaly:")
amp_sig = corr_df[
    (corr_df["apac_var"] == "Amplitude anomaly") &
    (corr_df["sig"] == "*")
].sort_values("r", key=abs, ascending=False)
print(amp_sig[["sector","index","r","p","n"]].head(10).to_string(index=False))

print(f"\n=== Done — outputs in {OUTPUT_DIR} ===")