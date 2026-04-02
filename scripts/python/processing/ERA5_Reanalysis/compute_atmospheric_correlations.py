"""
compute_atmospheric_correlations.py

Correlates APAC phase and amplitude anomalies with atmospheric indices:
  - Marshall SAM (monthly)
  - AAO (monthly)
  - TPI filtered (monthly)
  - ZW3 Raphael (annual)
  - ZW3 Goyal (annual: PC1, PC2, magnitude, phase)

Outputs:
  - correlations_annual.csv    — Pearson r and p-value, annual indices
  - correlations_seasonal.csv  — by season (DJF, MAM, JJA, SON)
  - correlation_heatmap.png    — summary figure

Reference period: 1979-2022 (common to all indices)
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
    "SIE_Weddell":                  "Weddell",
    "SIE_Amundsen_Bellingshausen":  "ABS",
    "SIE_Ross":                     "Ross",
    "SIE_East_Antarctica":          "East Antarctica",
    "SIE_King_Haakon":              "King Haakon",
}

# =============================================================================
# 1. LOAD APAC ANNUAL PARAMETERS
# =============================================================================

annual = pd.read_csv(ANNUAL_CSV)
annual = annual[annual["Year"].between(1979, 2022)]

# =============================================================================
# 2. LOAD ATMOSPHERIC INDICES
# =============================================================================

# --- 2a. Marshall SAM (wide format, header = month names) ---
sam_raw = pd.read_csv(
    os.path.join(INDICES_DIR, "marshall_sam_monthly.txt"),
    sep=r"\s+", header=0,
    names=["Year","Jan","Feb","Mar","Apr","May","Jun",
           "Jul","Aug","Sep","Oct","Nov","Dec"]
)
sam_raw = sam_raw[sam_raw["Year"].between(1979, 2022)]
sam_long = sam_raw.melt(id_vars="Year", var_name="month_name", value_name="SAM")
month_map = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,
             "Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}
sam_long["month"] = sam_long["month_name"].map(month_map)
sam_long = sam_long[["Year","month","SAM"]].sort_values(["Year","month"])

# Annual mean SAM
sam_annual = sam_long.groupby("Year")["SAM"].mean().reset_index()
sam_annual.columns = ["Year","SAM_annual"]

# Seasonal SAM
def seasonal_mean(df, val_col, season_months, new_col):
    sub = df[df["month"].isin(season_months)].groupby("Year")[val_col].mean().reset_index()
    sub.columns = ["Year", new_col]
    return sub

sam_djf = seasonal_mean(sam_long, "SAM", [12,1,2], "SAM_DJF")
sam_mam = seasonal_mean(sam_long, "SAM", [3,4,5],  "SAM_MAM")
sam_jja = seasonal_mean(sam_long, "SAM", [6,7,8],  "SAM_JJA")
sam_son = seasonal_mean(sam_long, "SAM", [9,10,11],"SAM_SON")

# --- 2b. AAO (same wide format) ---
aao_raw = pd.read_csv(
    os.path.join(INDICES_DIR, "daily_aao_sam.txt"),
    sep=r"\s+", header=0,
    names=["Year","Jan","Feb","Mar","Apr","May","Jun",
           "Jul","Aug","Sep","Oct","Nov","Dec"]
)
aao_raw = aao_raw[aao_raw["Year"].between(1979, 2022)]
aao_long = aao_raw.melt(id_vars="Year", var_name="month_name", value_name="AAO")
aao_long["month"] = aao_long["month_name"].map(month_map)
aao_long = aao_long[["Year","month","AAO"]].sort_values(["Year","month"])

aao_annual = aao_long.groupby("Year")["AAO"].mean().reset_index()
aao_annual.columns = ["Year","AAO_annual"]

aao_djf = seasonal_mean(aao_long, "AAO", [12,1,2], "AAO_DJF")
aao_mam = seasonal_mean(aao_long, "AAO", [3,4,5],  "AAO_MAM")
aao_jja = seasonal_mean(aao_long, "AAO", [6,7,8],  "AAO_JJA")
aao_son = seasonal_mean(aao_long, "AAO", [9,10,11],"AAO_SON")

# --- 2c. TPI filtered ---
tpi_rows = []
with open(os.path.join(INDICES_DIR, "tpi_filtered.txt")) as f:
    lines = f.readlines()
for line in lines[1:]:  # skip header row (date range)
    parts = line.split()
    if len(parts) != 13:
        continue
    year = int(parts[0])
    vals = [float(v) for v in parts[1:]]
    tpi_rows.append([year] + vals)

tpi_raw = pd.DataFrame(tpi_rows,
    columns=["Year","Jan","Feb","Mar","Apr","May","Jun",
             "Jul","Aug","Sep","Oct","Nov","Dec"])
# Replace missing
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

# --- 2d. ZW3 Raphael (already annual) ---
zw3r = pd.read_csv(os.path.join(INDICES_DIR, "ZW3_raphael_annual.csv"))
zw3r = zw3r[zw3r["year"].between(1979, 2022)]
zw3r.columns = ["Year","ZW3R_annual"]

# --- 2e. ZW3 Goyal (already annual) ---
zw3g = pd.read_csv(os.path.join(INDICES_DIR, "ZW3_goyal_annual.csv"))
zw3g = zw3g[zw3g["year"].between(1979, 2022)]
zw3g.columns = ["Year","ZW3G_PC1","ZW3G_PC2","ZW3G_magnitude","ZW3G_phase"]

# =============================================================================
# 3. BUILD MASTER INDEX TABLE
# =============================================================================

idx = sam_annual.copy()
for df in [aao_annual, tpi_annual, zw3r, zw3g]:
    idx = idx.merge(df, on="Year", how="left")

# Add seasonal indices
for df in [sam_djf, sam_mam, sam_jja, sam_son,
           aao_djf, aao_mam, aao_jja, aao_son,
           tpi_djf, tpi_mam, tpi_jja, tpi_son]:
    idx = idx.merge(df, on="Year", how="left")

print(f"Index table: {idx.shape}")
print(idx.head())

# =============================================================================
# 4. CORRELATIONS
# =============================================================================

INDEX_COLS_ANNUAL = [
    "SAM_annual", "AAO_annual", "TPI_annual",
    "ZW3R_annual", "ZW3G_magnitude", "ZW3G_phase",
    "SAM_DJF","SAM_MAM","SAM_JJA","SAM_SON",
    "AAO_DJF","AAO_MAM","AAO_JJA","AAO_SON",
    "TPI_DJF","TPI_MAM","TPI_JJA","TPI_SON",
]

APAC_VARS = ["max_doy_anom", "amplitude_anom"]
APAC_LABELS = {"max_doy_anom": "Phase anomaly", "amplitude_anom": "Amplitude anomaly"}

results = []

for sec_col, sec_label in SECTORS.items():
    sec_data = annual[annual["sector"] == sec_col].copy()

    for apac_var in APAC_VARS:
        for idx_col in INDEX_COLS_ANNUAL:
            merged = sec_data[["Year", apac_var]].merge(
                idx[["Year", idx_col]], on="Year", how="inner"
            ).dropna()

            if len(merged) < 10:
                continue

            r, p = stats.pearsonr(merged[idx_col], merged[apac_var])
            results.append({
                "sector":     sec_label,
                "apac_var":   APAC_LABELS[apac_var],
                "index":      idx_col,
                "r":          round(r, 3),
                "p":          round(p, 4),
                "n":          len(merged),
                "sig":        "*" if p < 0.05 else ("." if p < 0.10 else ""),
            })

corr_df = pd.DataFrame(results)
corr_df.to_csv(os.path.join(OUTPUT_DIR, "correlations_all.csv"), index=False)
print(f"\nCorrelations computed: {len(corr_df)} rows")

# =============================================================================
# 5. SUMMARY HEATMAP — annual indices only, phase anomaly
# =============================================================================

HEATMAP_INDICES = ["SAM_annual","AAO_annual","TPI_annual",
                   "ZW3R_annual","ZW3G_magnitude"]
HEATMAP_LABELS  = ["SAM","AAO","TPI","ZW3\n(Raphael)","ZW3\n(Goyal mag)"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
plt.rcParams.update({"font.family": "Nimbus Sans"})

for ax, apac_var, title in zip(
    axes,
    ["Phase anomaly", "Amplitude anomaly"],
    ["Phase anomaly vs atmospheric indices",
     "Amplitude anomaly vs atmospheric indices"]
):
    sub = corr_df[
        (corr_df["apac_var"] == apac_var) &
        (corr_df["index"].isin(HEATMAP_INDICES))
    ].pivot(index="sector", columns="index", values="r")

    # Reorder
    sector_order = ["Weddell","ABS","Ross","East Antarctica","King Haakon"]
    sub = sub.reindex(sector_order)[HEATMAP_INDICES]
    sub.columns = HEATMAP_LABELS

    im = ax.imshow(sub.values, cmap="RdBu_r", vmin=-0.6, vmax=0.6,
                   aspect="auto")

    ax.set_xticks(range(len(HEATMAP_LABELS)))
    ax.set_xticklabels(HEATMAP_LABELS, fontsize=11)
    ax.set_yticks(range(len(sector_order)))
    ax.set_yticklabels(sector_order, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

    # Annotate cells with r value and significance
    sig_sub = corr_df[
        (corr_df["apac_var"] == apac_var) &
        (corr_df["index"].isin(HEATMAP_INDICES))
    ].pivot(index="sector", columns="index", values="sig")
    sig_sub = sig_sub.reindex(sector_order)[HEATMAP_INDICES]
    sig_sub.columns = HEATMAP_LABELS

    for i in range(len(sector_order)):
        for j in range(len(HEATMAP_LABELS)):
            r_val = sub.values[i, j]
            sig   = sig_sub.values[i, j]
            if not np.isnan(r_val):
                ax.text(j, i, f"{r_val:.2f}{sig}",
                        ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        color="white" if abs(r_val) > 0.35 else "#2C2C2A",
                        path_effects=[pe.withStroke(linewidth=2,
                                      foreground="black" if abs(r_val) > 0.35
                                      else "white")])

plt.colorbar(im, ax=axes, label="Pearson r", shrink=0.8, pad=0.02)
fig.suptitle("APAC phase and amplitude anomalies vs atmospheric indices\n"
             "* p<0.05   . p<0.10   (annual means, 1979–2022)",
             fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap_annual.png"),
            dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Heatmap saved")

print(f"\n=== Done — outputs in {OUTPUT_DIR} ===")

# Quick preview of strongest correlations
print("\nTop 10 significant correlations (phase anomaly):")
phase_sig = corr_df[
    (corr_df["apac_var"] == "Phase anomaly") &
    (corr_df["sig"] == "*")
].sort_values("r", key=abs, ascending=False)
print(phase_sig[["sector","index","r","p","n"]].head(10).to_string(index=False))

print("\nTop 10 significant correlations (amplitude anomaly):")
amp_sig = corr_df[
    (corr_df["apac_var"] == "Amplitude anomaly") &
    (corr_df["sig"] == "*")
].sort_values("r", key=abs, ascending=False)
print(amp_sig[["sector","index","r","p","n"]].head(10).to_string(index=False))