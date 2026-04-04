"""
compute_atmospheric_correlations.py

Correlates APAC phase and amplitude anomalies with atmospheric indices:
  - Marshall SAM (monthly)
  - AAO (monthly)
  - ZW3 Raphael NEW (monthly netCDF, ERA5 1979-2025) — replaces old annual CSV
  - ZW3 Goyal (annual: PC1, PC2, magnitude, phase)
  - ASL Hosking v3 (monthly ERA5) — RelCenPres, SON and DJF
  - Annual mean SIE anomaly (1979-2010 baseline) — shows conflation effect

Both APAC variables, SIE anomaly, and atmospheric indices are linearly
detrended before correlation to remove spurious trend-driven signals.

Outputs:
  - correlations_all.csv            — Pearson r and p-value, all indices
  - correlation_heatmap_annual.png  — summary heatmap

Reference period: 1979-2022
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy import stats
from scipy.ndimage import uniform_filter1d
import netCDF4 as nc
import os

# =============================================================================
# 0. PATHS
# =============================================================================

INDICES_DIR = "/user/geog/falejandraperez/sea-ice-phase/data/indices/"
ANNUAL_CSV  = "/user/geog/falejandraperez/sea-ice-phase/scripts/R/chapter3/data/annual_params.csv"
DAILY_CSV   = "/user/geog/falejandraperez/sea-ice-phase/scripts/R/chapter3/data/daily_fitted.csv"
SIE_CSV     = "/user/geog/falejandraperez/sea-ice-phase/scripts/R/Sea_Ice_Sheets/SIE_daily_sector_and_circumpolar_million_km2.csv"
OUTPUT_DIR  = "/user/geog/falejandraperez/sea-ice-phase/scripts/R/chapter3/figures/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SECTORS = {
    "SIE_Weddell":                 "Weddell",
    "SIE_Amundsen_Bellingshausen": "ABS",
    "SIE_Ross":                    "Ross",
    "SIE_East_Antarctica":         "East Antarctica",
    "SIE_King_Haakon":             "King Haakon",
}

YEAR_MIN = 1979
YEAR_MAX = 2022

# =============================================================================
# 1. LOAD APAC ANNUAL PARAMETERS
# =============================================================================

annual = pd.read_csv(ANNUAL_CSV)
annual = annual[annual["Year"].between(YEAR_MIN, YEAR_MAX)]

# =============================================================================
# 1b. DERIVE TREND AND RESIDUAL ANNUAL INDICES FROM DAILY FITTED CSV
# =============================================================================
# From the sequential decomposition:
#   trend_component  — s(tdate) partial from gam_apac, in SIE units
#   est_anomaly      — smoothed residual after trend+amp+phase removed (GARCH)
#
# Annual indices derived:
#   trend_annual     — mean of trend_component for that year
#                      (captures where in the multi-decadal trend that year sits)
#   residual_std     — std dev of est_anomaly for that year
#                      (captures how much unexplained day-to-day noise there was)
#   residual_mean    — mean of est_anomaly for that year
#                      (signed: was the unexplained anomaly systematically + or -)

print("Loading daily fitted CSV for trend/residual indices...")
daily = pd.read_csv(DAILY_CSV, parse_dates=["Date"])
daily = daily[daily["Year"].between(YEAR_MIN, YEAR_MAX)]

daily_indices = []
for sec_col, sec_label in SECTORS.items():
    d = daily[daily["sector"] == sec_col].copy()

    ann = (d.groupby("Year")
            .agg(
                trend_annual  = ("trend_component", "mean"),
                residual_std  = ("est_anomaly",     "std"),
                residual_mean = ("est_anomaly",     "mean"),
            )
            .reset_index())
    ann["sector"]       = sec_col
    ann["sector_label"] = sec_label
    daily_indices.append(ann)

daily_idx = pd.concat(daily_indices)
print(f"  Daily indices computed: {daily_idx.shape}")
print(f"  Columns: {list(daily_idx.columns)}")



month_map = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,
             "Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}

def load_wide_index(filepath, val_col):
    """Load a year x month wide-format index file."""
    df = pd.read_csv(
        filepath, sep=r"\s+", header=0,
        names=["Year","Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]
    )
    df = df[df["Year"].between(YEAR_MIN, YEAR_MAX)]
    long = df.melt(id_vars="Year", var_name="month_name", value_name=val_col)
    long["month"] = long["month_name"].map(month_map)
    return long[["Year","month",val_col]].sort_values(["Year","month"])


def seasonal_mean(df, val_col, season_months, new_col):
    """Compute seasonal mean, handling DJF year alignment."""
    df = df.copy()
    if set(season_months) == {12, 1, 2}:
        df.loc[df["month"] == 12, "Year"] = df.loc[df["month"] == 12, "Year"] + 1
    sub = (df[df["month"].isin(season_months)]
           .groupby("Year")[val_col].mean().reset_index())
    sub.columns = ["Year", new_col]
    return sub


# --- Marshall SAM ---
sam_long    = load_wide_index(os.path.join(INDICES_DIR, "marshall_sam_monthly.txt"), "SAM")
sam_annual  = sam_long.groupby("Year")["SAM"].mean().reset_index()
sam_annual.columns = ["Year","SAM_annual"]
sam_djf     = seasonal_mean(sam_long, "SAM", [12,1,2],  "SAM_DJF")
sam_son     = seasonal_mean(sam_long, "SAM", [9,10,11], "SAM_SON")

# --- AAO ---
aao_long    = load_wide_index(os.path.join(INDICES_DIR, "daily_aao_sam.txt"), "AAO")
aao_annual  = aao_long.groupby("Year")["AAO"].mean().reset_index()
aao_annual.columns = ["Year","AAO_annual"]
aao_djf     = seasonal_mean(aao_long, "AAO", [12,1,2],  "AAO_DJF")
aao_son     = seasonal_mean(aao_long, "AAO", [9,10,11], "AAO_SON")

# --- ZW3 Raphael NEW — monthly netCDF ---
print("Loading new ZW3 Raphael netCDF...")
zw3_nc = nc.Dataset(os.path.join(INDICES_DIR, "zw3_monthly_index_ERA5_1979-2025.nc"))

# time is stored as days/hours since epoch — decode to dates
time_var  = zw3_nc.variables["time"]
time_vals = nc.num2date(time_var[:], units=time_var.units,
                        calendar=getattr(time_var, "calendar", "standard"))
zw3_vals  = zw3_nc.variables["zw3"][:]
zw3_nc.close()

zw3_new = pd.DataFrame({
    "date" : pd.to_datetime([str(t) for t in time_vals]),
    "ZW3R" : np.array(zw3_vals).flatten()
})
zw3_new["Year"]  = zw3_new["date"].dt.year
zw3_new["month"] = zw3_new["date"].dt.month
zw3_new = zw3_new[zw3_new["Year"].between(YEAR_MIN, YEAR_MAX)]

zw3r_long   = zw3_new[["Year","month","ZW3R"]]
zw3r_annual = zw3r_long.groupby("Year")["ZW3R"].mean().reset_index()
zw3r_annual.columns = ["Year","ZW3R_annual"]
zw3r_son    = seasonal_mean(zw3r_long, "ZW3R", [9,10,11], "ZW3R_SON")
zw3r_djf    = seasonal_mean(zw3r_long, "ZW3R", [12,1,2],  "ZW3R_DJF")

print(f"  ZW3R loaded: {len(zw3r_annual)} years, "
      f"{zw3r_annual['Year'].min()}–{zw3r_annual['Year'].max()}")

# --- ZW3 Goyal ---
zw3g = pd.read_csv(os.path.join(INDICES_DIR, "ZW3_goyal_annual.csv"))
zw3g = zw3g[zw3g["year"].between(YEAR_MIN, YEAR_MAX)]
zw3g.columns = ["Year","ZW3G_PC1","ZW3G_PC2","ZW3G_magnitude","ZW3G_phase"]

# --- ASL Hosking v3 — RelCenPres, SON and DJF ---
print("Loading ASL index...")
asl = pd.read_csv(os.path.join(INDICES_DIR, "asli_era5_v3-latest.csv"),
                  comment="#", parse_dates=["time"])
asl["Year"]  = asl["time"].dt.year
asl["month"] = asl["time"].dt.month
asl = asl[asl["Year"].between(YEAR_MIN, YEAR_MAX)]
asl_long = asl[["Year","month","RelCenPres"]].copy()
asl_long.columns = ["Year","month","ASL"]

asl_son = seasonal_mean(asl_long, "ASL", [9,10,11], "ASL_SON")
asl_djf = seasonal_mean(asl_long, "ASL", [12,1,2],  "ASL_DJF")

print(f"  ASL loaded: {asl_son['Year'].min()}–{asl_son['Year'].max()}")

# =============================================================================
# 3. COMPUTE ANNUAL MEAN SIE ANOMALY (1979-2010 baseline)
# Used to demonstrate conflation effect in heatmap
# =============================================================================
print("Computing annual mean SIE anomaly...")

sie = pd.read_csv(SIE_CSV)
sie["Date"] = pd.to_datetime(sie["Date"], format="%m/%d/%y")
sie["Year"] = sie["Date"].dt.year
sie["DOY"]  = sie["Date"].dt.dayofyear
sie = sie[sie["Year"].between(YEAR_MIN, YEAR_MAX)].sort_values("Date")

# 5-day smoothing then 1979-2010 climatology
for col in list(SECTORS.keys()) + ["SIE_circumpolar"]:
    sie[col] = uniform_filter1d(
        sie[col].fillna(method="ffill").values, size=5, mode="nearest")

clim = (sie[sie["Year"].between(1979, 2010)]
        .groupby("DOY")[list(SECTORS.keys()) + ["SIE_circumpolar"]].mean())

for col in list(SECTORS.keys()) + ["SIE_circumpolar"]:
    sie[f"{col}_anom"] = sie[col] - sie["DOY"].map(clim[col])

# Annual mean anomaly per sector
sie_annual_anom = (sie.groupby("Year")
                   [[f"{s}_anom" for s in SECTORS.keys()]]
                   .mean().reset_index())

# Rename to something clean
sie_annual_anom = sie_annual_anom.rename(columns={
    f"{s}_anom": f"SIE_anom_{SECTORS[s].replace(' ','_')}"
    for s in SECTORS.keys()
})

print(f"  SIE anomaly computed: {sie_annual_anom.shape}")

# =============================================================================
# 4. BUILD MASTER INDEX TABLE
# =============================================================================

idx = sam_annual.copy()
for df in [aao_annual,
           sam_son, sam_djf,
           aao_son, aao_djf,
           zw3r_annual, zw3r_son, zw3r_djf,
           zw3g,
           asl_son, asl_djf,
           sie_annual_anom]:
    idx = idx.merge(df, on="Year", how="left")

print(f"\nMaster index table: {idx.shape}")
print(f"  Columns: {list(idx.columns)}")

# =============================================================================
# 5. DETREND ALL SERIES
# =============================================================================

def detrend_series(df, year_col, val_col):
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

# All index columns to detrend
INDEX_COLS_ALL = [
    "SAM_annual", "SAM_SON", "SAM_DJF",
    "AAO_annual", "AAO_SON", "AAO_DJF",
    "ZW3R_annual", "ZW3R_SON", "ZW3R_DJF",
    "ZW3G_PC1", "ZW3G_PC2", "ZW3G_magnitude", "ZW3G_phase",
    "ASL_SON", "ASL_DJF",
] + [f"SIE_anom_{SECTORS[s].replace(' ','_')}" for s in SECTORS.keys()]

idx_dt = idx.copy()
# Cast all index columns to float64 to avoid dtype mismatch FutureWarning
for col in INDEX_COLS_ALL:
    if col in idx_dt.columns:
        idx_dt[col] = idx_dt[col].astype("float64")
for col in INDEX_COLS_ALL:
    if col in idx_dt.columns:
        tmp = detrend_series(idx_dt[["Year", col]].dropna(), "Year", col)
        idx_dt.loc[tmp.index, col] = tmp[col].values.astype("float64")

print("Detrending complete")

# =============================================================================
# 6. CORRELATIONS
# =============================================================================

APAC_VARS = ["max_doy_anom", "amplitude_anom"]
APAC_LABELS = {
    "max_doy_anom":   "Phase anomaly",
    "amplitude_anom": "Amplitude anomaly",
}

# Also correlate SIE anomaly with indices (to show conflation)
results = []

for sec_col, sec_label in SECTORS.items():
    sec_data = annual_dt[annual_dt["sector"] == sec_col].copy()
    sie_col  = f"SIE_anom_{sec_label.replace(' ','_')}"

    # Merge SIE anomaly
    sie_sec = idx_dt[["Year", sie_col]].copy()
    sie_sec = detrend_series(sie_sec, "Year", sie_col)

    # --- Phase and amplitude anomaly correlations ---
    for apac_var in APAC_VARS:
        for idx_col in INDEX_COLS_ALL:
            if "SIE_anom" in idx_col:
                continue
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

    # --- SIE anomaly correlations ---
    for idx_col in INDEX_COLS_ALL:
        if "SIE_anom" in idx_col:
            continue
        merged = sie_sec[["Year", sie_col]].merge(
            idx_dt[["Year", idx_col]], on="Year", how="inner"
        ).dropna()
        if len(merged) < 10:
            continue
        r, p = stats.pearsonr(merged[idx_col], merged[sie_col])
        results.append({
            "sector":   sec_label,
            "apac_var": "SIE anomaly",
            "index":    idx_col,
            "r":        round(r, 3),
            "p":        round(p, 4),
            "n":        len(merged),
            "sig":      "*" if p < 0.05 else ("." if p < 0.10 else ""),
        })

    # --- Trend and residual correlations (from daily_fitted) ---
    sec_daily = daily_idx[daily_idx["sector"] == sec_col].copy()

    for daily_var, daily_label in [
        ("trend_annual",  "Trend (annual mean)"),
        ("residual_mean", "Residual (annual mean)"),
        ("residual_std",  "Residual (annual std dev)"),
    ]:
        # Detrend the daily-derived variable
        sec_daily_dt = detrend_series(sec_daily[["Year", daily_var]].dropna(),
                                      "Year", daily_var)
        for idx_col in INDEX_COLS_ALL:
            if "SIE_anom" in idx_col:
                continue
            merged = sec_daily_dt[["Year", daily_var]].merge(
                idx_dt[["Year", idx_col]], on="Year", how="inner"
            ).dropna()
            if len(merged) < 10:
                continue
            r, p = stats.pearsonr(merged[idx_col], merged[daily_var])
            results.append({
                "sector":   sec_label,
                "apac_var": daily_label,
                "index":    idx_col,
                "r":        round(r, 3),
                "p":        round(p, 4),
                "n":        len(merged),
                "sig":      "*" if p < 0.05 else ("." if p < 0.10 else ""),
            })

corr_df = pd.DataFrame(results)
corr_df.to_csv(os.path.join(OUTPUT_DIR, "correlations_all.csv"), index=False)
print(f"\nCorrelations computed: {len(corr_df)} rows")

# =============================================================================
# 7. HEATMAP — 3 panels: SIE anomaly | Phase | Amplitude
# Trimmed to 6 indices. Colorbar below plots. 2016/2023 overlay dots.
# =============================================================================

plt.rcParams.update({"font.family": "Nimbus Sans"})

HEATMAP_INDICES = [
    "SAM_annual", "SAM_SON", "SAM_DJF",
    "ZW3R_annual",
    "ASL_SON", "ASL_DJF",
]
HEATMAP_LABELS = [
    "SAM\nannual", "SAM\nSON", "SAM\nDJF",
    "ZW3\nRaphael",
    "ASL\nSON", "ASL\nDJF",
]

sector_order = ["Weddell", "ABS", "Ross", "East Antarctica", "King Haakon"]

panel_vars   = ["SIE anomaly", "Phase anomaly", "Amplitude anomaly"]
panel_titles = [
    "Raw SIE anomaly\nvs atmospheric indices",
    "Phase anomaly (max DOY)\nvs atmospheric indices",
    "Amplitude anomaly\nvs atmospheric indices",
]

# --- Compute normalised index anomalies for 2016 and 2023 overlay -----------
# For each index, compute the z-score of 2016 and 2023 relative to full record
# We'll overlay a marker on cells where the index was notably anomalous
# (|z| > 1.0) in that year AND the correlation is significant

overlay_years = {2016: "#D85A30", 2023: "#BA7517"}

index_zscores = {}
for idx_col in HEATMAP_INDICES:
    if idx_col not in idx.columns:
        continue
    s = idx.set_index("Year")[idx_col].dropna()
    mu, sd = s.mean(), s.std()
    if sd > 0:
        index_zscores[idx_col] = {yr: (s.get(yr, np.nan) - mu) / sd
                                  for yr in overlay_years}

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

for ax, apac_var, title in zip(axes, panel_vars, panel_titles):
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
    ax.set_title("")  # titles added in Keynote

    # r values and significance
    for i in range(len(sector_order)):
        for j in range(len(HEATMAP_LABELS)):
            r_val = sub.values[i, j]
            sig   = sig_sub.values[i, j]
            if not np.isnan(float(r_val)):
                ax.text(j, i, f"{float(r_val):.2f}{sig}",
                        ha="center", va="center",
                        fontsize=8, fontweight="bold",
                        color="white" if abs(float(r_val)) > 0.35
                              else "#2C2C2A",
                        path_effects=[pe.withStroke(
                            linewidth=2,
                            foreground="black" if abs(float(r_val)) > 0.35
                                       else "white")])

    # 2016 / 2023 overlay dots
    for j, (idx_col, idx_label) in enumerate(zip(HEATMAP_INDICES, HEATMAP_LABELS)):
        for yr, col in overlay_years.items():
            z = index_zscores.get(idx_col, {}).get(yr, np.nan)
            if np.isnan(z) or abs(z) < 0.8:
                continue
            for i, sec in enumerate(sector_order):
                cell_sig = sig_sub.values[i, j]
                if cell_sig in ["*", "."]:
                    yoff = -0.32 if yr == 2016 else 0.32
                    ax.plot(j + 0.35, i + yoff, "o",
                            color=col, markersize=8,
                            markeredgecolor="white", markeredgewidth=0.8,
                            zorder=5, clip_on=False)

# Divider between SIE and Phase panels
axes[0].spines["right"].set_linewidth(2.0)
axes[0].spines["right"].set_color("#888780")

# Colorbar — below the plots, properly positioned
fig.subplots_adjust(bottom=0.22, top=0.88, left=0.06,
                    right=0.97, wspace=0.06)
cbar = fig.colorbar(im, ax=axes,
                    orientation="horizontal",
                    fraction=0.03, pad=0.18, aspect=40,
                    label="Pearson r")
cbar.ax.tick_params(labelsize=10)

# Legend for overlay dots
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#D85A30",
           markersize=7, label="2016 index anomalous (|z|>0.8)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#BA7517",
           markersize=7, label="2023 index anomalous (|z|>0.8)"),
]
fig.legend(handles=legend_elements, loc="lower center",
           ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.01),
           frameon=False)

fig.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap_annual.png"),
            dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Heatmap saved")

# =============================================================================
# 7b. EXPLORATORY HEATMAP — trend and residual components
# For your own peace of mind — not for the talk
# =============================================================================
print("Saving exploratory heatmap (trend + residual)...")

EXPLORE_VARS   = ["Trend (annual mean)", "Residual (annual mean)", "Residual (annual std dev)"]
EXPLORE_TITLES = [
    "Trend component\nvs atmospheric indices",
    "Residual anomaly (mean)\nvs atmospheric indices",
    "Residual anomaly (std dev)\nvs atmospheric indices",
]

fig, axes = plt.subplots(1, 3, figsize=(22, 5), sharey=True)

for ax, apac_var, title in zip(axes, EXPLORE_VARS, EXPLORE_TITLES):
    sub = corr_df[
        (corr_df["apac_var"] == apac_var) &
        (corr_df["index"].isin(HEATMAP_INDICES))
    ].pivot(index="sector", columns="index", values="r")

    if sub.empty:
        ax.set_title(f"{title}\n(no data)", fontsize=10)
        continue

    sub = sub.reindex(sector_order)[HEATMAP_INDICES]
    sub.columns = HEATMAP_LABELS

    sig_sub = corr_df[
        (corr_df["apac_var"] == apac_var) &
        (corr_df["index"].isin(HEATMAP_INDICES))
    ].pivot(index="sector", columns="index", values="sig")
    sig_sub = sig_sub.reindex(sector_order)[HEATMAP_INDICES]
    sig_sub.columns = HEATMAP_LABELS

    im2 = ax.imshow(sub.values.astype(float),
                    cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")

    ax.set_xticks(range(len(HEATMAP_LABELS)))
    ax.set_xticklabels(HEATMAP_LABELS, fontsize=9)
    ax.set_yticks(range(len(sector_order)))
    ax.set_yticklabels(sector_order, fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)

    for i in range(len(sector_order)):
        for j in range(len(HEATMAP_LABELS)):
            r_val = sub.values[i, j]
            sig   = sig_sub.values[i, j]
            if not np.isnan(float(r_val)):
                ax.text(j, i, f"{float(r_val):.2f}{sig}",
                        ha="center", va="center",
                        fontsize=8, fontweight="bold",
                        color="white" if abs(float(r_val)) > 0.35
                              else "#2C2C2A",
                        path_effects=[pe.withStroke(
                            linewidth=2,
                            foreground="black" if abs(float(r_val)) > 0.35
                                       else "white")])

cbar2 = fig.colorbar(im2, ax=axes, label="Pearson r",
                     orientation="horizontal",
                     shrink=0.35, pad=0.22, aspect=30)
cbar2.ax.tick_params(labelsize=10)

fig.suptitle(
    "Trend and residual components vs atmospheric indices — 1979–2022\n"
    "* p<0.05   . p<0.10   |   exploratory — not for talk",
    fontsize=11, y=1.02
)
fig.subplots_adjust(bottom=0.25, top=0.88, left=0.04,
                    right=0.97, wspace=0.08)
fig.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap_exploratory.png"),
            dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Exploratory heatmap saved")

# =============================================================================
# 8. TERMINAL PREVIEW
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

print("\nTop 10 significant correlations — SIE anomaly (conflated):")
sie_sig = corr_df[
    (corr_df["apac_var"] == "SIE anomaly") &
    (corr_df["sig"] == "*")
].sort_values("r", key=abs, ascending=False)
print(sie_sig[["sector","index","r","p","n"]].head(10).to_string(index=False))

print("\nTop significant correlations — trend component:")
trend_sig = corr_df[
    (corr_df["apac_var"] == "Trend (annual mean)") &
    (corr_df["sig"] == "*")
].sort_values("r", key=abs, ascending=False)
print(trend_sig[["sector","index","r","p","n"]].head(10).to_string(index=False))

print("\nTop significant correlations — residual mean:")
res_sig = corr_df[
    (corr_df["apac_var"] == "Residual (annual mean)") &
    (corr_df["sig"] == "*")
].sort_values("r", key=abs, ascending=False)
print(res_sig[["sector","index","r","p","n"]].head(10).to_string(index=False))

print(f"\n=== Done — outputs in {OUTPUT_DIR} ===")

# =============================================================================
# SUPPLEMENTARY 1 — Full heatmap: all 5 decomposition components, all indices
# =============================================================================
print("\nSupplementary 1: Full heatmap (all components, all indices)...")

SUPP_INDICES = [
    "SAM_annual", "SAM_SON", "SAM_DJF",
    "AAO_annual", "AAO_SON", "AAO_DJF",
    "ZW3R_annual", "ZW3R_SON", "ZW3R_DJF",
    "ZW3G_PC2", "ZW3G_magnitude",
    "ASL_SON", "ASL_DJF",
]
SUPP_LABELS = [
    "SAM\nannual", "SAM\nSON", "SAM\nDJF",
    "AAO\nannual", "AAO\nSON", "AAO\nDJF",
    "ZW3R\nannual", "ZW3R\nSON", "ZW3R\nDJF",
    "ZW3G\nPC2", "ZW3G\nmag",
    "ASL\nSON", "ASL\nDJF",
]

supp_vars = [
    "SIE anomaly",
    "Phase anomaly",
    "Amplitude anomaly",
    "Trend (annual mean)",
    "Residual (annual mean)",
]
supp_titles = [
    "Raw SIE anomaly",
    "Phase anomaly",
    "Amplitude anomaly",
    "Trend component",
    "Residual (mean)",
]

fig, axes = plt.subplots(1, 5, figsize=(32, 5), sharey=True)

for ax, apac_var, title in zip(axes, supp_vars, supp_titles):
    sub = corr_df[
        (corr_df["apac_var"] == apac_var) &
        (corr_df["index"].isin(SUPP_INDICES))
    ].pivot(index="sector", columns="index", values="r")

    if sub.empty:
        ax.set_title(title + "\n(no data)", fontsize=9)
        continue

    sub = sub.reindex(sector_order)[SUPP_INDICES]
    sub.columns = SUPP_LABELS

    sig_sub = corr_df[
        (corr_df["apac_var"] == apac_var) &
        (corr_df["index"].isin(SUPP_INDICES))
    ].pivot(index="sector", columns="index", values="sig")
    sig_sub = sig_sub.reindex(sector_order)[SUPP_INDICES]
    sig_sub.columns = SUPP_LABELS

    im_s = ax.imshow(sub.values.astype(float),
                     cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")

    ax.set_xticks(range(len(SUPP_LABELS)))
    ax.set_xticklabels(SUPP_LABELS, fontsize=7)
    ax.set_yticks(range(len(sector_order)))
    ax.set_yticklabels(sector_order, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=8)

    for i in range(len(sector_order)):
        for j in range(len(SUPP_LABELS)):
            r_val = sub.values[i, j]
            sig   = sig_sub.values[i, j]
            if not np.isnan(float(r_val)):
                ax.text(j, i, f"{float(r_val):.2f}{sig}",
                        ha="center", va="center",
                        fontsize=6.5, fontweight="bold",
                        color="white" if abs(float(r_val)) > 0.35
                              else "#2C2C2A",
                        path_effects=[pe.withStroke(
                            linewidth=1.5,
                            foreground="black" if abs(float(r_val)) > 0.35
                                       else "white")])

fig.subplots_adjust(bottom=0.22, top=0.88, left=0.04,
                    right=0.97, wspace=0.08)
cbar_s = fig.colorbar(im_s, ax=axes,
                      orientation="horizontal",
                      fraction=0.03, pad=0.18, aspect=50,
                      label="Pearson r")
cbar_s.ax.tick_params(labelsize=9)

fig.suptitle(
    "All decomposition components vs atmospheric indices — 1979–2022  "
    "|  * p<0.05   . p<0.10   |  all series linearly detrended  "
    "|  SUPPLEMENTARY",
    fontsize=9, y=1.0
)
fig.savefig(os.path.join(OUTPUT_DIR, "supp_heatmap_full.png"),
            dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("  -> supp_heatmap_full.png")

# =============================================================================
# SUPPLEMENTARY 2 — Atmospheric index anomalies in 2016 and 2023
# Time series per index with 1979-2015 mean ± 1 std dev band
# 2016 and 2023 highlighted as coloured dots
# =============================================================================
print("Supplementary 2: Atmospheric index anomalies in 2016 and 2023...")

SUPP2_INDICES = [
    "SAM_annual", "SAM_SON",
    "ZW3R_annual", "ZW3R_SON",
    "ASL_SON",     "ASL_DJF",
]
SUPP2_LABELS = [
    "SAM annual", "SAM SON",
    "ZW3 Raphael annual", "ZW3 Raphael SON",
    "ASL SON", "ASL DJF",
]

n_idx = len(SUPP2_INDICES)
fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey=False)
axes = axes.flatten()

for ax, idx_col, idx_label in zip(axes, SUPP2_INDICES, SUPP2_LABELS):
    if idx_col not in idx.columns:
        ax.set_visible(False)
        continue

    s = idx.set_index("Year")[idx_col].dropna().sort_index()
    yrs = s.index.values

    # Baseline: 1979-2015
    base = s[s.index <= 2015]
    mu   = base.mean()
    sd   = base.std()

    # Time series
    ax.plot(yrs, s.values, color="#888780", lw=1.0, zorder=2)

    # Mean and ±1 std band
    ax.axhline(mu, color="#2C2C2A", lw=1.0, ls="--", zorder=3,
               label="1979–2015 mean")
    ax.axhspan(mu - sd, mu + sd, color="#B4B2A9", alpha=0.2, zorder=1,
               label="±1 std dev")

    # 2016 and 2023 dots
    for yr, col, marker in [(2016, "#D85A30", "o"), (2023, "#BA7517", "D")]:
        if yr in s.index:
            ax.scatter(yr, s[yr], color=col, s=80, zorder=5,
                       marker=marker, edgecolors="white", linewidth=0.8,
                       label=str(yr))
            # Annotate z-score
            z = (s[yr] - mu) / sd
            ax.annotate(f"z={z:+.1f}",
                        xy=(yr, s[yr]),
                        xytext=(6, 6), textcoords="offset points",
                        fontsize=8, color=col, fontweight="bold",
                        path_effects=[pe.withStroke(linewidth=2,
                                                     foreground="white")])

    ax.set_title(idx_label, fontsize=11, fontweight="bold")
    ax.set_xlim(1978, 2024)
    ax.tick_params(labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if ax == axes[0]:
        ax.legend(fontsize=8, loc="upper right", frameon=False)

fig.suptitle(
    "Atmospheric index values — 2016 and 2023 in context\n"
    "Dashed line = 1979–2015 mean   |   shading = ±1 std dev   "
    "|   z-scores relative to 1979–2015 baseline   |   SUPPLEMENTARY",
    fontsize=10, y=1.01
)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "supp_index_anomalies_2016_2023.png"),
            dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("  -> supp_index_anomalies_2016_2023.png")

print(f"\n=== All supplementary figures saved to {OUTPUT_DIR} ===")