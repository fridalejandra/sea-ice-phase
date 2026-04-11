"""
compute_atmospheric_correlations.py

Correlates APAC phase and amplitude anomalies with atmospheric indices:
  - Marshall SAM (monthly)
  - AAO (monthly)
  - ZW3 Raphael NEW (monthly netCDF, ERA5 1979-2025)
  - ZW3 Goyal (annual: PC1, PC2, magnitude, phase)
  - ASL Hosking v3 (monthly ERA5) — RelCenPres, SON and DJF
  - Annual mean SIE anomaly (1979-2010 baseline)

All series linearly detrended before correlation.

Outputs:
  - correlations_all.csv
  - correlation_heatmap_annual.png       — Phase + Amplitude stacked, z-score bar chart
  - correlation_heatmap_exploratory.png  — Trend + Residual exploratory
  - supp_heatmap_full.png                — All 5 components, grid layout
  - supp_index_anomalies_2016_2023.png   — Index time series context

Reference period: 1979-2023
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
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
YEAR_MAX = 2023

# =============================================================================
# 1. LOAD APAC ANNUAL PARAMETERS
# =============================================================================

annual = pd.read_csv(ANNUAL_CSV)
annual = annual[annual["Year"].between(YEAR_MIN, YEAR_MAX)]

# =============================================================================
# 1b. TREND AND RESIDUAL INDICES FROM DAILY FITTED CSV
# =============================================================================

print("Loading daily fitted CSV...")
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
print(f"  Daily indices: {daily_idx.shape}")

# =============================================================================
# 2. LOAD ATMOSPHERIC INDICES
# =============================================================================

month_map = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,
             "Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}

def load_wide_index(filepath, val_col):
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
    df = df.copy()
    if set(season_months) == {12, 1, 2}:
        df.loc[df["month"] == 12, "Year"] = df.loc[df["month"] == 12, "Year"] + 1
    sub = (df[df["month"].isin(season_months)]
           .groupby("Year")[val_col].mean().reset_index())
    sub.columns = ["Year", new_col]
    return sub


# SAM
sam_long   = load_wide_index(os.path.join(INDICES_DIR, "marshall_sam_monthly.txt"), "SAM")
sam_annual = sam_long.groupby("Year")["SAM"].mean().reset_index()
sam_annual.columns = ["Year","SAM_annual"]
sam_djf    = seasonal_mean(sam_long, "SAM", [12,1,2],  "SAM_DJF")
sam_son    = seasonal_mean(sam_long, "SAM", [9,10,11], "SAM_SON")

# AAO
aao_long   = load_wide_index(os.path.join(INDICES_DIR, "daily_aao_sam.txt"), "AAO")
aao_annual = aao_long.groupby("Year")["AAO"].mean().reset_index()
aao_annual.columns = ["Year","AAO_annual"]
aao_djf    = seasonal_mean(aao_long, "AAO", [12,1,2],  "AAO_DJF")
aao_son    = seasonal_mean(aao_long, "AAO", [9,10,11], "AAO_SON")

# ZW3 Raphael
print("Loading ZW3 Raphael netCDF...")
zw3_nc    = nc.Dataset(os.path.join(INDICES_DIR, "zw3_monthly_index_ERA5_1979-2025.nc"))
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
print(f"  ZW3R: {zw3r_annual['Year'].min()}–{zw3r_annual['Year'].max()}")

# ZW3 Goyal
zw3g = pd.read_csv(os.path.join(INDICES_DIR, "ZW3_goyal_annual.csv"))
zw3g = zw3g[zw3g["year"].between(YEAR_MIN, YEAR_MAX)]
zw3g.columns = ["Year","ZW3G_PC1","ZW3G_PC2","ZW3G_magnitude","ZW3G_phase"]

# ASL
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
print(f"  ASL: {asl_son['Year'].min()}–{asl_son['Year'].max()}")

# Niño3.4 (ENSO)
# Download from: https://psl.noaa.gov/data/correlation/nina34.data
# Format: year | jan feb mar apr may jun jul aug sep oct nov dec
import urllib.request

nino_path = os.path.join(INDICES_DIR, "nina34.data")
if not os.path.exists(nino_path):
    print("Downloading Niño3.4 data...")
    urllib.request.urlretrieve(
        "https://psl.noaa.gov/data/correlation/nina34.data",
        nino_path
    )
    print(f"  Saved to {nino_path}")
else:
    print(f"  Niño3.4 file already exists: {nino_path}")

print("Loading Niño3.4...")
nino_path = os.path.join(INDICES_DIR, "nina34.data")
print("Loading Niño3.4...")
nino_rows = []
with open(nino_path, "r") as f:
    for line in f:
        parts = line.split()
        # skip header line (only 2 elements) and short lines
        if len(parts) != 13:
            continue
        try:
            year = int(parts[0])
        except ValueError:
            continue
        for month, val in enumerate(parts[1:], start=1):
            nino_rows.append({"Year": year, "month": month, "NINO34": float(val)})

nino_long = pd.DataFrame(nino_rows)
nino_long = nino_long[nino_long["NINO34"] > -9]
nino_long = nino_long[nino_long["Year"].between(YEAR_MIN, YEAR_MAX)]

nino_annual = nino_long.groupby("Year")["NINO34"].mean().reset_index()
nino_annual.columns = ["Year", "NINO34_annual"]
nino_djf    = seasonal_mean(nino_long, "NINO34", [12,1,2],  "NINO34_DJF")
nino_mam    = seasonal_mean(nino_long, "NINO34", [3,4,5],   "NINO34_MAM")
nino_son    = seasonal_mean(nino_long, "NINO34", [9,10,11], "NINO34_SON")
print(f"  Niño3.4: {nino_annual['Year'].min()}–{nino_annual['Year'].max()}")

# =============================================================================
# 3. SIE ANOMALY (1979-2010 baseline)
# =============================================================================
print("Computing SIE anomaly...")

sie = pd.read_csv(SIE_CSV)
sie["Date"] = pd.to_datetime(sie["Date"], format="%m/%d/%y")
sie["Year"] = sie["Date"].dt.year
sie["DOY"]  = sie["Date"].dt.dayofyear
sie = sie[sie["Year"].between(YEAR_MIN, YEAR_MAX)].sort_values("Date")

for col in list(SECTORS.keys()) + ["SIE_circumpolar"]:
    sie[col] = uniform_filter1d(
        sie[col].fillna(method="ffill").values, size=5, mode="nearest")

clim = (sie[sie["Year"].between(1979, 2010)]
        .groupby("DOY")[list(SECTORS.keys()) + ["SIE_circumpolar"]].mean())

for col in list(SECTORS.keys()) + ["SIE_circumpolar"]:
    sie[f"{col}_anom"] = sie[col] - sie["DOY"].map(clim[col])

sie_annual_anom = (sie.groupby("Year")
                   [[f"{s}_anom" for s in SECTORS.keys()]]
                   .mean().reset_index())
sie_annual_anom = sie_annual_anom.rename(columns={
    f"{s}_anom": f"SIE_anom_{SECTORS[s].replace(' ','_')}"
    for s in SECTORS.keys()
})

# =============================================================================
# 4. MASTER INDEX TABLE
# =============================================================================

idx = sam_annual.copy()
for df in [aao_annual,
           sam_son, sam_djf,
           aao_son, aao_djf,
           zw3r_annual, zw3r_son, zw3r_djf,
           zw3g,
           asl_son, asl_djf,
           nino_annual, nino_djf, nino_mam, nino_son,
           sie_annual_anom]:
    idx = idx.merge(df, on="Year", how="left")

print(f"\nMaster index table: {idx.shape}")

# =============================================================================
# 5. DETREND
# =============================================================================
# WHY DETREND?
# Many climate indices share a common long-term linear trend — SAM has trended
# positive, Antarctic SIE trended positive then negative. If we don't remove
# these trends, we would find spurious correlations that simply reflect the
# shared long-term drift rather than genuine year-to-year variability.
# Detrending isolates the interannual variability we actually want to explain.
#
# HOW IT WORKS:
# For each series we fit a straight line (y = slope * year + intercept) using
# ordinary least squares (stats.linregress). We then subtract that line from
# the data. What remains is the residual variability around the trend — the
# interannual signal.
#
# ASSUMPTION: the trend is linear. If the trend is non-linear (e.g. a sudden
# shift in 2016) linear detrending may not fully remove it. This is a known
# limitation given the post-2016 regime shift.

def detrend_series(df, year_col, val_col):
    df = df.copy().dropna(subset=[val_col]).sort_values(year_col)
    if len(df) < 5:
        return df
    x = df[year_col].values.astype(float)
    y = df[val_col].values.astype(float)
    # Fit a linear trend: y = slope * x + intercept
    slope, intercept, _, _, _ = stats.linregress(x, y)
    # Subtract the fitted trend from the data
    df[val_col] = y - (slope * x + intercept)
    return df


# Detrend APAC phase and amplitude anomalies for each sector
annual_dt = []
for sec_col in SECTORS.keys():
    sec = annual[annual["sector"] == sec_col].copy()
    for var in ["max_doy_anom", "amplitude_anom"]:
        sec = detrend_series(sec, "Year", var)
    annual_dt.append(sec)
annual_dt = pd.concat(annual_dt)

# List of all atmospheric index columns to detrend
INDEX_COLS_ALL = [
    "SAM_annual", "SAM_SON", "SAM_DJF",        # Southern Annular Mode seasonal means
    "AAO_annual", "AAO_SON", "AAO_DJF",         # Antarctic Oscillation (closely related to SAM)
    "ZW3R_annual", "ZW3R_SON", "ZW3R_DJF",      # Zonal Wave 3 — Raphael index
    "ZW3G_PC1", "ZW3G_PC2", "ZW3G_magnitude", "ZW3G_phase",  # ZW3 — Goyal EOF decomposition
    "ASL_SON", "ASL_DJF",                        # Amundsen Sea Low relative central pressure
    "NINO34_annual", "NINO34_DJF", "NINO34_MAM", "NINO34_SON",  # ENSO Niño3.4 SST
] + [f"SIE_anom_{SECTORS[s].replace(' ','_')}" for s in SECTORS.keys()]

# Detrend each index column in the master index table
idx_dt = idx.copy()
for col in INDEX_COLS_ALL:
    if col in idx_dt.columns:
        idx_dt[col] = idx_dt[col].astype("float64")
for col in INDEX_COLS_ALL:
    if col in idx_dt.columns:
        tmp = detrend_series(idx_dt[["Year", col]].dropna(), "Year", col)
        idx_dt.loc[tmp.index, col] = tmp[col].values.astype("float64")

# Save the detrended master index table so it can be loaded by other scripts
idx_dt.to_csv(os.path.join(OUTPUT_DIR, "master_index_detrended.csv"), index=False)
print("Detrending complete")
print(f"  Detrended index table saved to {OUTPUT_DIR}master_index_detrended.csv")

# =============================================================================
# EFFECTIVE DEGREES OF FREEDOM
# =============================================================================
# WHY THIS MATTERS:
# Pearson r significance is tested assuming each year is an independent
# observation. With n=44 years, the standard formula gives ~42 degrees of
# freedom (df = n - 2). However, climate variables often show serial
# autocorrelation — the value in year t is correlated with year t+1 because
# the ocean and atmosphere have memory. This means consecutive years are not
# fully independent, and the true effective degrees of freedom (n_eff) is
# lower than 44. Using n instead of n_eff produces p-values that are too
# small (overconfident significance).
#
# HOW WE ESTIMATE n_eff:
# We use the Chelton (1983) formula, which accounts for autocorrelation
# at all lags:
#   1/n_eff = (1/n) * sum_{k=-(n-1)}^{n-1} rho_x(k) * rho_y(k)
# where rho_x(k) and rho_y(k) are the autocorrelation functions of x and y
# at lag k. In practice we truncate at a reasonable number of lags (here 5
# years given annual data).
#
# The adjusted p-value uses a t-distribution with n_eff - 2 degrees of
# freedom instead of n - 2. This gives a more honest (conservative)
# significance estimate.

def effective_n(x, y, max_lag=5):
    """
    Estimate effective sample size accounting for serial autocorrelation
    in both x and y using the Chelton (1983) approach.

    Parameters:
        x, y    : 1D numpy arrays of the same length (detrended, no NaNs)
        max_lag : maximum lag to include in the autocorrelation sum

    Returns:
        n_eff   : effective sample size (float, minimum 3)
    """
    n = len(x)
    # Normalise both series to zero mean unit variance
    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()

    # Compute autocorrelation at each lag k for both series
    # np.correlate gives the full cross-correlation; we normalise by n
    acf_x = np.correlate(x, x, mode="full") / n
    acf_y = np.correlate(y, y, mode="full") / n

    # The zero-lag is at index n-1 in the full output
    mid = n - 1

    # Sum autocorrelation products from lag 1 to max_lag
    # (lag 0 contributes 1.0 * 1.0 = 1.0, already in the 1/n term)
    acf_sum = 1.0  # lag 0 contribution
    for k in range(1, min(max_lag + 1, n)):
        acf_sum += 2.0 * acf_x[mid + k] * acf_y[mid + k]

    # n_eff = n / acf_sum, clipped to a minimum of 3
    n_eff = n / acf_sum
    return max(3.0, float(n_eff))


def pearsonr_adjusted(x, y, max_lag=5):
    """
    Pearson correlation with p-value adjusted for serial autocorrelation.

    Returns:
        r       : Pearson correlation coefficient
        p_adj   : p-value using effective degrees of freedom
        n_eff   : effective sample size used for the test
    """
    r, _ = stats.pearsonr(x, y)
    n_eff = effective_n(x, y, max_lag=max_lag)
    # t-statistic using effective df
    # Formula: t = r * sqrt(n_eff - 2) / sqrt(1 - r^2)
    df_eff = n_eff - 2
    t_stat = r * np.sqrt(df_eff) / np.sqrt(1 - r**2 + 1e-12)
    # Two-tailed p-value from t-distribution with df_eff degrees of freedom
    p_adj = 2 * stats.t.sf(abs(t_stat), df=df_eff)
    return r, round(p_adj, 4), round(n_eff, 1)


# =============================================================================
# 6. CORRELATIONS
# =============================================================================
# WHAT WE ARE DOING:
# For each combination of (sector, APAC variable, atmospheric index) we compute
# the Pearson correlation coefficient r and its p-value over 1979-2023.
#
# Pearson r measures the strength and direction of a LINEAR relationship
# between two variables. It ranges from -1 (perfect negative) to +1 (perfect
# positive). r=0 means no linear relationship.
#
# The p-value answers: if there were truly NO relationship between these two
# variables, what is the probability of observing an r this large just by
# chance? p < 0.05 means less than 5% chance of a false positive.
#
# We report TWO p-values for each correlation:
#   p        : standard p-value assuming n independent observations
#   p_adj    : adjusted p-value using effective degrees of freedom
#              (more conservative, accounts for serial autocorrelation)
#
# STRUCTURE OF THE LOOP:
# Outer loop  → each of 5 sectors
# Middle loop → each APAC variable (phase anomaly, amplitude anomaly)
# Inner loop  → each atmospheric index
# For each combination: merge on Year, drop NaNs, compute r and p

APAC_LABELS = {
    "max_doy_anom":   "Phase anomaly",
    "amplitude_anom": "Amplitude anomaly",
}

results = []

for sec_col, sec_label in SECTORS.items():
    sec_data = annual_dt[annual_dt["sector"] == sec_col].copy()
    sie_col  = f"SIE_anom_{sec_label.replace(' ','_')}"
    sie_sec  = detrend_series(idx_dt[["Year", sie_col]].copy(), "Year", sie_col)

    # --- Phase and amplitude anomaly correlations ---
    # For each APAC variable (phase, amplitude) correlate with each index
    for apac_var, apac_label in APAC_LABELS.items():
        for idx_col in INDEX_COLS_ALL:
            if "SIE_anom" in idx_col:
                continue  # SIE anomaly handled separately below
            # Merge APAC data with index data on Year — inner join keeps
            # only years where both variables have valid data
            merged = sec_data[["Year", apac_var]].merge(
                idx_dt[["Year", idx_col]], on="Year", how="inner"
            ).dropna()
            if len(merged) < 10:
                continue  # skip if too few data points for a meaningful test
            x = merged[idx_col].values
            y = merged[apac_var].values
            # Standard Pearson r and p-value (assumes independence)
            r, p = stats.pearsonr(x, y)
            # Adjusted p-value accounting for serial autocorrelation
            _, p_adj, n_eff = pearsonr_adjusted(x, y)
            results.append({
                "sector":   sec_label,
                "apac_var": apac_label,
                "index":    idx_col,
                "r":        round(r, 3),
                "p":        round(p, 4),
                "p_adj":    p_adj,       # adjusted for autocorrelation
                "n_eff":    n_eff,        # effective sample size
                "n":        len(merged),  # actual sample size
                "sig":      "*" if p < 0.05 else ("." if p < 0.10 else ""),
                "sig_adj":  "*" if p_adj < 0.05 else ("." if p_adj < 0.10 else ""),
            })

    # --- SIE anomaly correlations ---
    # Raw annual mean SIE anomaly vs each index — comparable to Raphael & Hobbs (2014)
    # Note: SIE anomaly contains both phase and amplitude signals mixed together
    for idx_col in INDEX_COLS_ALL:
        if "SIE_anom" in idx_col:
            continue
        merged = sie_sec[["Year", sie_col]].merge(
            idx_dt[["Year", idx_col]], on="Year", how="inner"
        ).dropna()
        if len(merged) < 10:
            continue
        x = merged[idx_col].values
        y = merged[sie_col].values
        r, p = stats.pearsonr(x, y)
        _, p_adj, n_eff = pearsonr_adjusted(x, y)
        results.append({
            "sector":   sec_label,
            "apac_var": "SIE anomaly",
            "index":    idx_col,
            "r":        round(r, 3),
            "p":        round(p, 4),
            "p_adj":    p_adj,
            "n_eff":    n_eff,
            "n":        len(merged),
            "sig":      "*" if p < 0.05 else ("." if p < 0.10 else ""),
            "sig_adj":  "*" if p_adj < 0.05 else ("." if p_adj < 0.10 else ""),
        })

    # --- Trend and residual correlations ---
    # These correlate the annual mean and std dev of the APAC daily residuals
    # with each index — exploratory analysis of what drives longer-term
    # variability and short-term scatter around the seasonal cycle
    sec_daily = daily_idx[daily_idx["sector"] == sec_col].copy()
    for daily_var, daily_label in [
        ("trend_annual",  "Trend (annual mean)"),
        ("residual_mean", "Residual (annual mean)"),
        ("residual_std",  "Residual (annual std dev)"),
    ]:
        sec_dt = detrend_series(sec_daily[["Year", daily_var]].dropna(),
                                "Year", daily_var)
        for idx_col in INDEX_COLS_ALL:
            if "SIE_anom" in idx_col:
                continue
            merged = sec_dt[["Year", daily_var]].merge(
                idx_dt[["Year", idx_col]], on="Year", how="inner"
            ).dropna()
            if len(merged) < 10:
                continue
            x = merged[idx_col].values
            y = merged[daily_var].values
            r, p = stats.pearsonr(x, y)
            _, p_adj, n_eff = pearsonr_adjusted(x, y)
            results.append({
                "sector":   sec_label,
                "apac_var": daily_label,
                "index":    idx_col,
                "r":        round(r, 3),
                "p":        round(p, 4),
                "p_adj":    p_adj,
                "n_eff":    n_eff,
                "n":        len(merged),
                "sig":      "*" if p < 0.05 else ("." if p < 0.10 else ""),
                "sig_adj":  "*" if p_adj < 0.05 else ("." if p_adj < 0.10 else ""),
            })

corr_df = pd.DataFrame(results)

# =============================================================================
# FDR CORRECTION (Benjamini-Hochberg)
# =============================================================================
# WHY WE NEED THIS:
# When running many hypothesis tests simultaneously, we expect some to appear
# significant purely by chance. With 570 tests at p < 0.05, we'd expect
# ~28 false positives even if there were no real relationships.
#
# The False Discovery Rate (FDR) method controls the expected proportion of
# false positives among all significant results. The Benjamini-Hochberg (BH)
# procedure is the standard approach:
#   1. Sort all p-values from smallest to largest
#   2. Find the largest k such that p(k) <= (k/m) * alpha
#      where m = total number of tests and alpha = 0.05
#   3. Declare all tests up to rank k as significant
#
# We apply FDR correction to BOTH the standard p-values and the
# autocorrelation-adjusted p-values (p_adj).
#
# INTERPRETATION:
# If nothing survives FDR correction it does NOT mean there are no real
# relationships — it means the signal-to-noise ratio across the full test
# set is low. The physically consistent results (SAM-EA amplitude r=0.47,
# ZW3-Ross amplitude r=-0.41) are still meaningful; FDR correction tells
# us to be cautious about marginal results.

from statsmodels.stats.multitest import multipletests

# FDR on standard p-values
pvals = corr_df["p"].values
_, pvals_fdr, _, _ = multipletests(pvals, method="fdr_bh")
corr_df["p_fdr"]   = pvals_fdr.round(4)
corr_df["sig_fdr"] = corr_df["p_fdr"].apply(
    lambda p: "*" if p < 0.05 else ("." if p < 0.10 else ""))

# FDR on autocorrelation-adjusted p-values
pvals_adj = corr_df["p_adj"].values
_, pvals_adj_fdr, _, _ = multipletests(pvals_adj, method="fdr_bh")
corr_df["p_adj_fdr"]   = pvals_adj_fdr.round(4)
corr_df["sig_adj_fdr"] = corr_df["p_adj_fdr"].apply(
    lambda p: "*" if p < 0.05 else ("." if p < 0.10 else ""))

corr_df.to_csv(os.path.join(OUTPUT_DIR, "correlations_all.csv"), index=False)
print(f"\nCorrelations computed: {len(corr_df)} rows")
print(f"Significant uncorrected (p<0.05):              {(corr_df['sig']=='*').sum()}")
print(f"Significant adjusted for autocorr (p_adj<0.05): {(corr_df['sig_adj']=='*').sum()}")
print(f"Significant FDR corrected (p_fdr<0.05):         {(corr_df['sig_fdr']=='*').sum()}")
print(f"Significant FDR + autocorr (p_adj_fdr<0.05):    {(corr_df['sig_adj_fdr']=='*').sum()}")

# Print effective n summary for key results
print("\n=== Effective sample size for key correlations ===")
key_results = corr_df[
    (corr_df["apac_var"].isin(["Phase anomaly","Amplitude anomaly"])) &
    (corr_df["sig"] == "*")
].sort_values("p")[["sector","apac_var","index","r","p","p_adj","n","n_eff","sig","sig_adj"]].head(15)
print(key_results.to_string(index=False))

# =============================================================================
# NIÑO3.4 SUMMARY — correlations with phase and amplitude, plus 2016/2023 z-scores
# =============================================================================

print("\n=== Niño3.4 correlations (Phase + Amplitude, uncorrected) ===")
nino_cols = ["NINO34_annual", "NINO34_DJF", "NINO34_MAM", "NINO34_SON"]
nino_corrs = corr_df[
    (corr_df["index"].isin(nino_cols)) &
    (corr_df["apac_var"].isin(["Phase anomaly", "Amplitude anomaly"]))
].sort_values("p")
print(nino_corrs[["sector","apac_var","index","r","p","sig","p_fdr","sig_fdr","n"]].to_string(index=False))

sig_nino_raw = nino_corrs[nino_corrs["sig"] == "*"]
sig_nino_fdr = nino_corrs[nino_corrs["sig_fdr"] == "*"]
print(f"\n  Significant uncorrected (p<0.05): {len(sig_nino_raw)}")
print(f"  Significant FDR-corrected (p_fdr<0.05): {len(sig_nino_fdr)}")
if len(sig_nino_fdr) == 0:
    print("  → No Niño3.4 correlations survive FDR correction")

print("\n=== Niño3.4 z-scores: 2016 and 2023 (relative to 1979–2015) ===")
for col in nino_cols:
    if col not in idx.columns:
        continue
    s    = idx.set_index("Year")[col].dropna()
    base = s[s.index <= 2015]
    mu, sd = base.mean(), base.std()
    z2016 = (s.get(2016, float("nan")) - mu) / sd
    z2023 = (s.get(2023, float("nan")) - mu) / sd
    print(f"  {col:<20} 2016: {z2016:+.2f}σ   2023: {z2023:+.2f}σ")

# =============================================================================
# FULL SIGNIFICANT RESULTS TABLE
# =============================================================================

print("\n=== All significant correlations — uncorrected p<0.05 ===")
sig_all = corr_df[corr_df["sig"] == "*"].sort_values("p")
print(sig_all[["sector","apac_var","index","r","p","sig","p_fdr","sig_fdr","n"]].to_string(index=False))

print("\n=== All significant correlations — FDR corrected p_fdr<0.05 ===")
sig_fdr_all = corr_df[corr_df["sig_fdr"] == "*"].sort_values("p_fdr")
print(sig_fdr_all[["sector","apac_var","index","r","p","sig","p_fdr","sig_fdr","n"]].to_string(index=False))

print("\n=== Summary by index group ===")
for grp, cols in [
    ("SAM",    ["SAM_annual","SAM_SON","SAM_DJF"]),
    ("ZW3",    ["ZW3R_annual","ZW3R_SON","ZW3R_DJF"]),
    ("ASL",    ["ASL_SON","ASL_DJF"]),
    ("NINO34", ["NINO34_annual","NINO34_DJF","NINO34_MAM","NINO34_SON"]),
    ("ZW3G",   ["ZW3G_PC1","ZW3G_PC2","ZW3G_magnitude","ZW3G_phase"]),
]:
    sub = corr_df[corr_df["index"].isin(cols)]
    n_raw = (sub["sig"] == "*").sum()
    n_fdr = (sub["sig_fdr"] == "*").sum()
    print(f"  {grp:<10} uncorrected: {n_raw:2d}   FDR: {n_fdr:2d}")

# =============================================================================
# 7. MAIN HEATMAP — Phase (top) + Amplitude (bottom) stacked | Z-score bar (right)
# =============================================================================

plt.rcParams.update({"font.family": "Nimbus Sans"})

HEATMAP_INDICES = [
    "SAM_annual", "SAM_SON", "SAM_DJF",
    "ZW3R_annual",
    "ASL_SON", "ASL_DJF",
]
HEATMAP_LABELS = [
    "SAM\nannual", "SAM\nSON", "SAM\nDJF",
    "ZW3\nannual",
    "ASL\nSON", "ASL\nDJF",
]
HEATMAP_LABELS_SHORT = [
    "SAM annual", "SAM SON", "SAM DJF",
    "ZW3 annual",
    "ASL SON", "ASL DJF",
]

sector_order = ["Weddell", "ABS", "Ross", "East Antarctica", "King Haakon"]

panel_vars   = ["Phase anomaly", "Amplitude anomaly"]
panel_titles = [
    "Phase anomaly\nvs atmospheric indices",
    "Amplitude anomaly\nvs atmospheric indices",
]

# Z-scores for 2016 and 2023 relative to 1979-2015 baseline
OVERLAY_YEARS = [2016, 2023]
index_zscores = {}
for idx_col in HEATMAP_INDICES:
    if idx_col not in idx.columns:
        continue
    s    = idx.set_index("Year")[idx_col].dropna()
    base = s[s.index <= 2015]
    mu, sd = base.mean(), base.std()
    if sd > 0:
        index_zscores[idx_col] = {yr: (s.get(yr, np.nan) - mu) / sd
                                  for yr in OVERLAY_YEARS}

# --- Figure: 2-row left column (heatmaps) + 1-row right column (bar chart) ---
fig = plt.figure(figsize=(16, 9))

# Outer gridspec: 2 columns — left heatmaps, right bar chart
gs_outer = fig.add_gridspec(1, 2, width_ratios=[2.8, 1.4],
                             left=0.08, right=0.97,
                             bottom=0.12, top=0.93,
                             wspace=0.30)

# Inner gridspec for the two stacked heatmaps
gs_left = gs_outer[0].subgridspec(2, 1, hspace=0.55)
ax_phase = fig.add_subplot(gs_left[0])
ax_amp   = fig.add_subplot(gs_left[1])

# Right: z-score bar chart
ax_bar = fig.add_subplot(gs_outer[1])

# --- Draw heatmaps ---
im = None
for ax, apac_var, title in zip([ax_phase, ax_amp], panel_vars, panel_titles):
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
    ax.set_yticklabels(sector_order, fontsize=11)   # always show on both panels
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)

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

# Force y-tick labels visible on both panels regardless of gridspec sharing
for ax in [ax_phase, ax_amp]:
    plt.setp(ax.get_yticklabels(), visible=True)

# Shared colorbar below both heatmaps
cbar = fig.colorbar(im, ax=[ax_phase, ax_amp],
                    orientation="horizontal",
                    fraction=0.03, pad=0.08, aspect=40,
                    shrink=0.85)
cbar.ax.tick_params(labelsize=10)
cbar.set_label("Pearson r  |  * p<0.05   . p<0.10", fontsize=10)

# --- Z-score bar chart ---
z_2016 = [index_zscores.get(c, {}).get(2016, np.nan) for c in HEATMAP_INDICES]
z_2023 = [index_zscores.get(c, {}).get(2023, np.nan) for c in HEATMAP_INDICES]

n      = len(HEATMAP_INDICES)
y_pos  = np.arange(n)
height = 0.35

bars_2016 = ax_bar.barh(y_pos + height/2, z_2016, height,
                         color="#D85A30", alpha=0.9,
                         label="2016", zorder=3)
bars_2023 = ax_bar.barh(y_pos - height/2, z_2023, height,
                         color="#FFD700", alpha=0.9,
                         edgecolor="#2C2C2A", linewidth=0.5,
                         label="2023", zorder=3)

ax_bar.axvline( 1.0, color="#2C2C2A", lw=1.0, ls="--", alpha=0.5, zorder=2)
ax_bar.axvline(-1.0, color="#2C2C2A", lw=1.0, ls="--", alpha=0.5, zorder=2)
ax_bar.axvline( 0.0, color="#2C2C2A", lw=0.6, ls="-",  alpha=0.3, zorder=2)

ax_bar.set_yticks(y_pos)
ax_bar.set_yticklabels(HEATMAP_LABELS_SHORT, fontsize=10)
ax_bar.set_xlabel("z-score\n(relative to 1979–2015 mean)", fontsize=10)
ax_bar.set_title("Index anomaly\nin 2016 vs 2023", fontsize=12,
                 fontweight="bold", pad=10)
ax_bar.set_xlim(-3, 3)
ax_bar.spines["top"].set_visible(False)
ax_bar.spines["right"].set_visible(False)
ax_bar.legend(fontsize=10, loc="lower right", frameon=False)

for bar, z in zip(bars_2016, z_2016):
    if not np.isnan(z):
        xpos = z + (0.08 if z >= 0 else -0.08)
        ha   = "left" if z >= 0 else "right"
        ax_bar.text(xpos, bar.get_y() + bar.get_height()/2,
                    f"{z:+.1f}", va="center", ha=ha,
                    fontsize=8, color="#D85A30", fontweight="bold")

for bar, z in zip(bars_2023, z_2023):
    if not np.isnan(z):
        xpos = z + (0.08 if z >= 0 else -0.08)
        ha   = "left" if z >= 0 else "right"
        ax_bar.text(xpos, bar.get_y() + bar.get_height()/2,
                    f"{z:+.1f}", va="center", ha=ha,
                    fontsize=8, color="#8B7000", fontweight="bold")

fig.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap_annual.png"),
            dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Main heatmap saved")

# --- Goyal ZW3 correlation summary (terminal only) ---
print("\n=== Goyal ZW3 correlations (Phase + Amplitude) ===")
goyal_cols = ["ZW3G_PC1", "ZW3G_PC2", "ZW3G_magnitude", "ZW3G_phase"]
goyal_corrs = corr_df[
    (corr_df["index"].isin(goyal_cols)) &
    (corr_df["apac_var"].isin(["Phase anomaly", "Amplitude anomaly"]))
].sort_values("p")
print(goyal_corrs[["sector","apac_var","index","r","p","sig","n"]].to_string(index=False))
sig_goyal = goyal_corrs[goyal_corrs["sig"] != ""]
print(f"\n  Significant (p<0.10): {len(sig_goyal)} correlations")
if len(sig_goyal) == 0:
    print("  -> No Goyal ZW3 correlations reached p<0.10 for phase or amplitude")

# =============================================================================
# 7b. EXPLORATORY HEATMAP — Trend + Residual
# =============================================================================
print("Saving exploratory heatmap...")

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
    "Trend and residual components vs atmospheric indices — 1979–2023\n"
    "* p<0.05   . p<0.10   |   exploratory — not for talk",
    fontsize=11, y=1.02
)
fig.subplots_adjust(bottom=0.25, top=0.88, left=0.04, right=0.97, wspace=0.08)
fig.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap_exploratory.png"),
            dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Exploratory heatmap saved")

# =============================================================================
# 8. TERMINAL PREVIEW
# =============================================================================

print("\nAll significant correlations (p<0.05):")
all_sig = corr_df[corr_df["sig"] == "*"].sort_values("r", key=abs, ascending=False)
print(all_sig[["sector","apac_var","index","r","p"]].to_string(index=False))

for label in ["Phase anomaly", "Amplitude anomaly", "SIE anomaly",
              "Trend (annual mean)", "Residual (annual mean)"]:
    sub = corr_df[(corr_df["apac_var"] == label) & (corr_df["sig"] == "*")]
    sub = sub.sort_values("r", key=abs, ascending=False)
    print(f"\nTop significant — {label}:")
    print(sub[["sector","index","r","p","n"]].head(10).to_string(index=False))

print(f"\n=== Done — outputs in {OUTPUT_DIR} ===")

# =============================================================================
# SUPPLEMENTARY 1 — Full heatmap: all 5 components, grid layout (3 top, 2 bottom)
# =============================================================================
print("\nSupplementary 1: Full heatmap...")

SUPP_INDICES = [
    "SAM_annual", "SAM_SON", "SAM_DJF",
    "ZW3R_annual", "ZW3R_SON", "ZW3R_DJF",
    "ZW3G_PC2", "ZW3G_magnitude",
    "ASL_SON", "ASL_DJF",
]
SUPP_LABELS = [
    "SAM\nannual", "SAM\nSON", "SAM\nDJF",
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

# Grid layout: 3 panels top row, 2 panels bottom row
# Use constrained_layout=False and manage spacing manually for reliable colorbar
fig = plt.figure(figsize=(28, 12))

# Build a 2-row x 3-col gridspec with extra bottom space for the colorbar
gs_supp = fig.add_gridspec(
    2, 3,
    left=0.05, right=0.97,
    top=0.91, bottom=0.14,     # generous bottom margin for colorbar
    hspace=0.45, wspace=0.12
)

ax_top  = [fig.add_subplot(gs_supp[0, i]) for i in range(3)]
ax_bot0 = fig.add_subplot(gs_supp[1, 0])
ax_bot1 = fig.add_subplot(gs_supp[1, 1])
# Leave gs_supp[1, 2] empty (no axis created)

all_axes = ax_top + [ax_bot0, ax_bot1]

im_s = None
for ax, apac_var, title in zip(all_axes, supp_vars, supp_titles):
    sub = corr_df[
        (corr_df["apac_var"] == apac_var) &
        (corr_df["index"].isin(SUPP_INDICES))
    ].pivot(index="sector", columns="index", values="r")

    if sub.empty:
        ax.set_title(title + "\n(no data)", fontsize=9)
        ax.set_visible(False)
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
    ax.set_xticklabels(SUPP_LABELS, fontsize=8)
    ax.set_yticks(range(len(sector_order)))
    ax.set_yticklabels(sector_order, fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)

    for i in range(len(sector_order)):
        for j in range(len(SUPP_LABELS)):
            r_val = sub.values[i, j]
            sig   = sig_sub.values[i, j]
            if not np.isnan(float(r_val)):
                ax.text(j, i, f"{float(r_val):.2f}{sig}",
                        ha="center", va="center",
                        fontsize=7, fontweight="bold",
                        color="white" if abs(float(r_val)) > 0.35
                              else "#2C2C2A",
                        path_effects=[pe.withStroke(
                            linewidth=1.5,
                            foreground="black" if abs(float(r_val)) > 0.35
                                       else "white")])

# Colorbar: placed explicitly in figure coordinates so it sits
# cleanly below all panels regardless of gridspec quirks
if im_s is not None:
    cbar_ax = fig.add_axes([0.20, 0.05, 0.60, 0.025])  # [left, bottom, width, height]
    cbar_s  = fig.colorbar(im_s, cax=cbar_ax, orientation="horizontal")
    cbar_s.ax.tick_params(labelsize=10)
    cbar_s.set_label("Pearson r  |  * p<0.05   . p<0.10", fontsize=10)

fig.suptitle(
    "All decomposition components vs atmospheric indices — 1979–2023  |  SUPPLEMENTARY",
    fontsize=12, y=0.97
)
fig.savefig(os.path.join(OUTPUT_DIR, "supp_heatmap_full.png"),
            dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("  -> supp_heatmap_full.png")

# =============================================================================
# SUPPLEMENTARY 2 — Index anomalies 2016 and 2023
# =============================================================================
print("Supplementary 2: Index anomalies...")

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

fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey=False)
axes = axes.flatten()

for ax, idx_col, idx_label in zip(axes, SUPP2_INDICES, SUPP2_LABELS):
    if idx_col not in idx.columns:
        ax.set_visible(False)
        continue

    s    = idx.set_index("Year")[idx_col].dropna().sort_index()
    yrs  = s.index.values
    base = s[s.index <= 2015]
    mu   = base.mean()
    sd   = base.std()

    ax.plot(yrs, s.values, color="#888780", lw=1.0, zorder=2)
    ax.axhline(mu, color="#2C2C2A", lw=1.0, ls="--", zorder=3,
               label="1979–2015 mean")
    ax.axhspan(mu - sd, mu + sd, color="#B4B2A9", alpha=0.2, zorder=1,
               label="±1 std dev")

    for yr, col, marker in [(2016, "#D85A30", "o"), (2023, "#FFD700", "D")]:
        if yr in s.index:
            ax.scatter(yr, s[yr], color=col, s=100, zorder=5,
                       marker=marker, edgecolors="#2C2C2A", linewidth=0.8,
                       label=str(yr))
            z = (s[yr] - mu) / sd
            ax.annotate(f"z={z:+.1f}",
                        xy=(yr, s[yr]),
                        xytext=(6, 6), textcoords="offset points",
                        fontsize=9, color=col, fontweight="bold",
                        path_effects=[pe.withStroke(linewidth=2,
                                                    foreground="white")])

    ax.set_title(idx_label, fontsize=11, fontweight="bold")
    ax.set_xlim(1978, 2025)
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

print(f"\n=== All figures saved to {OUTPUT_DIR} ===")