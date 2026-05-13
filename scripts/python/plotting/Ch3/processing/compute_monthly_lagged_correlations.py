"""
Monthly lagged correlations between atmospheric indices and APAC phase/amplitude.

This script asks a more granular version of the main correlations question:
not just "does SAM correlate with annual amplitude" but "which month's SAM
best predicts the annual amplitude, and how far in advance?"

Three analyses in one script because they share the same data loading and
detrending infrastructure:

    1. Monthly cross-correlations
       Each month's index value (Jan–Dec) correlated against the annual
       APAC scalar for that year. Tells you when during the year the
       atmospheric state matters most for the ice outcome.

    2. Atmospheric autocorrelation structure
       Lag-1 through lag-12 autocorrelation for each index.
       Critical context — if SAM is strongly autocorrelated at lag 3, a
       lag-3 cross-correlation with ice might just be SAM's own persistence
       rather than a genuine 3-month predictive window.

    3. Partial correlations at peak lags
       For the strongest monthly cross-correlations, the partial correlation
       controlling for the index's lag-1 autocorrelation. This confirms
       whether the predictive signal is real or just persistence.

    4. Block bootstrap confidence intervals
       For all monthly cross-correlations, 95% CIs using the stationary
       block bootstrap (arch package). Block length set to 3 years by default
       — captures the typical decorrelation timescale of annual climate indices.
       Addresses the reviewer request for bootstrap CIs that respect
       temporal dependence.

    5. Permutation test for significance
       10,000 random permutations of the ice scalar to build the null
       distribution of r under no relationship. P-values from this
       distribution are more defensible than t-distribution p-values
       with small n and autocorrelated data.

    6. Leave-one-out sensitivity
       For the key pairs (EA amplitude ~ SAM, Ross amplitude ~ ZW3,
       ABS phase ~ ASL), compute r leaving each year out in turn.
       The resulting distribution shows how sensitive the result is
       to individual years — particularly the 2023 outlier.

Outputs (all saved to DATA_DIR):
    monthly_cross_correlations.csv  — r, bootstrap CI, permutation p per month/pair
    atmospheric_autocorrelations.csv — lag autocorrelations per index
    partial_correlations.csv        — partial r at peak lags for key pairs
    loo_sensitivity.csv             — leave-one-out r values for key pairs

Needs:
    master_index_detrended.csv      — from compute_atmospheric_correlations.py
    annual_params.csv               — APAC annual scalars
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore")

# Block bootstrap — requires arch package (pip install arch)
try:
    from arch.bootstrap import StationaryBootstrap
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False
    print("Warning: arch package not found. Block bootstrap will be skipped.")
    print("Install with: pip install arch")

DATA_DIR   = "/user/geog/falejandraperez/sea-ice-phase/scripts/R/Ch3/data"
INDEX_DIR  = "/user/geog/falejandraperez/sea-ice-phase/data/indices"

ANNUAL_CSV = os.path.join(DATA_DIR, "annual_params.csv")
INDEX_CSV  = os.path.join(DATA_DIR, "master_index_detrended.csv")

YEAR_MIN = 1979
YEAR_MAX = 2023

# Block bootstrap settings
BLOCK_SIZE   = 3     # years — typical decorrelation timescale for annual indices
N_BOOTSTRAP  = 5000  # replications
N_PERMUTE    = 10000 # permutation test replications
ALPHA        = 0.05  # significance level

SECTORS = {
    "SIE_Weddell"                : "Weddell",
    "SIE_Amundsen_Bellingshausen": "ABS",
    "SIE_Ross"                   : "Ross",
    "SIE_East_Antarctica"        : "East Antarctica",
    "SIE_King_Haakon"            : "King Haakon",
}

# Key pairs for LOO and partial correlation — the physically motivated ones
KEY_PAIRS = [
    ("SIE_East_Antarctica",         "amplitude_anom", "SAM"),
    ("SIE_Ross",                    "amplitude_anom", "ZW3R"),
    ("SIE_Amundsen_Bellingshausen", "max_doy_anom",   "ASL"),
    ("SIE_East_Antarctica",         "amplitude_anom", "Nino34"),
]

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]

APAC_VARS = {
    "max_doy_anom"  : "phase",
    "amplitude_anom": "amplitude",
}


# --- Load -----------------------------------------------------------------

print("Loading data...")
annual = pd.read_csv(ANNUAL_CSV)
annual = annual[annual["Year"].between(YEAR_MIN, YEAR_MAX)]

for col in ["max_doy_anom", "amplitude_anom"]:
    annual[col] = pd.to_numeric(annual[col], errors="coerce")

idx = pd.read_csv(INDEX_CSV)
idx = idx[idx["Year"].between(YEAR_MIN, YEAR_MAX)]

print(f"  Annual params: {len(annual)} rows")
print(f"  Index table:   {idx.shape}")


# --- Load raw monthly indices for month-by-month correlations -------------
# We need month-resolution index values, not just seasonal means.
# Load the raw monthly files directly.

print("\nLoading monthly index files...")

month_map = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,
             "Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}

# SAM
sam_raw = pd.read_csv(
    os.path.join(INDEX_DIR, "marshall_sam_monthly.txt"),
    delim_whitespace=True, header=0,
    names=["year","Jan","Feb","Mar","Apr","May","Jun",
           "Jul","Aug","Sep","Oct","Nov","Dec"]
)
sam_long = sam_raw.melt(id_vars="year", var_name="month_str", value_name="SAM")
sam_long["month"] = sam_long["month_str"].map(month_map)
sam_long = sam_long.dropna(subset=["SAM"]).drop(columns="month_str")

# ZW3 Raphael — annual only, so monthly not available. Use annual value
# repeated for each month as a placeholder — flag this in output.
zw3_ann = pd.read_csv(os.path.join(INDEX_DIR, "ZW3_raphael_annual.csv"))
zw3_ann.columns = ["year", "ZW3R"]
zw3_ann["year"] = zw3_ann["year"].astype(int)

# ZW3 Goyal monthly — use this for monthly ZW3
zw3_monthly = pd.read_csv(os.path.join(INDEX_DIR, "ZW3_raphael_monthly.csv"))
zw3_monthly = zw3_monthly[["year", "month", "ZW3_index"]].rename(
    columns={"ZW3_index": "ZW3R"})

# ZW3 Goyal monthly — supplementary index, magnitude component
zw3_goyal = pd.read_csv(os.path.join(INDEX_DIR, "ZW3_goyal_monthly.csv"))
zw3_goyal = zw3_goyal.rename(columns={"ZW3_magnitude": "ZW3G"})
zw3_goyal = zw3_goyal[["year", "month", "ZW3G"]].dropna()
# ASL monthly
asl_raw = pd.read_csv(
    os.path.join(INDEX_DIR, "asli_era5_v3-latest.csv"), comment="#")
asl_raw["time"]  = pd.to_datetime(asl_raw["time"])
asl_raw["year"]  = asl_raw["time"].dt.year
asl_raw["month"] = asl_raw["time"].dt.month
asl_raw = asl_raw.rename(columns={"RelCenPres": "ASL"})[["year","month","ASL"]]

# Nino34 monthly
nino_raw = pd.read_csv(
    os.path.join(INDEX_DIR, "nina34.data"),
    sep=r'\s+', skiprows=1, header=None,
    names=["year","Jan","Feb","Mar","Apr","May","Jun",
           "Jul","Aug","Sep","Oct","Nov","Dec"],
    na_values=["-99.99", -99.99],
    engine="python",
)
nino_raw = nino_raw[pd.to_numeric(nino_raw["year"], errors="coerce")
                    .between(1900, 2100)]
nino_raw["year"] = nino_raw["year"].astype(float).astype(int)
nino_long = nino_raw.melt(id_vars="year", var_name="month_str",
                           value_name="Nino34")
nino_long["month"] = nino_long["month_str"].map(month_map)
nino_long = nino_long.dropna(subset=["Nino34"]).drop(columns="month_str")

# Merge all monthly indices into one long table
monthly_idx = (sam_long
               .merge(asl_raw,      on=["year","month"], how="outer")
               .merge(nino_long,    on=["year","month"], how="outer")
               .merge(zw3_monthly[["year","month","ZW3R"]],
                      on=["year","month"], how="left")
               .merge(zw3_goyal[["year","month","ZW3G"]],
                      on=["year","month"], how="left"))

# Add ZW3G if monthly is available
if "ZW3G" in zw3_monthly.columns:
    monthly_idx = monthly_idx.merge(
        zw3_monthly[["year","month","ZW3G"]], on=["year","month"], how="left")

monthly_idx = monthly_idx[
    monthly_idx["year"].between(YEAR_MIN, YEAR_MAX)].sort_values(
    ["year","month"]).reset_index(drop=True)

# Detrend each monthly index column
for col in ["SAM","ASL","Nino34","ZW3G"]:
    if col not in monthly_idx.columns:
        continue
    for m in range(1, 13):
        mask = monthly_idx["month"] == m
        vals = monthly_idx.loc[mask, col].values.astype(float)
        yrs  = monthly_idx.loc[mask, "year"].values.astype(float)
        valid = ~np.isnan(vals)
        if valid.sum() > 5:
            s, i, *_ = stats.linregress(yrs[valid], vals[valid])
            monthly_idx.loc[mask, col] = vals - (s * yrs + i)

print(f"  Monthly index table: {monthly_idx.shape}")

INDEX_COLS_MONTHLY = [c for c in ["SAM","ASL","Nino34","ZW3R","ZW3G"]
                      if c in monthly_idx.columns]


# --- Detrend APAC annual scalars ------------------------------------------

def detrend_series(years, values):
    mask = ~np.isnan(values)
    if mask.sum() < 5:
        return values
    s, i, *_ = stats.linregress(years[mask].astype(float), values[mask])
    return values - (s * years.astype(float) + i)


annual_dt = []
for sec_col in SECTORS:
    sec = annual[annual["sector"] == sec_col].copy().sort_values("Year")
    x   = sec["Year"].values.astype(float)
    for var in APAC_VARS:
        if var in sec.columns:
            sec[var] = detrend_series(x, sec[var].values.astype(float))
    annual_dt.append(sec)
annual_dt = pd.concat(annual_dt)


# --- Helper: block bootstrap CI -------------------------------------------

def bootstrap_ci(x, y, block_size=BLOCK_SIZE, n_boot=N_BOOTSTRAP,
                 alpha=ALPHA):
    """
    95% CI for Pearson r using stationary block bootstrap.
    Returns (ci_lower, ci_upper, boot_r_std).
    Falls back to Fisher z-transform CI if arch is not available.
    """
    if len(x) < 8:
        return np.nan, np.nan, np.nan

    if HAS_ARCH:
        try:
            bs = StationaryBootstrap(block_size, x, y)
            boot_r = bs.apply(
                lambda args, kwargs: pearsonr(args[0], args[1])[0],
                n_boot
            )
            ci_l = np.percentile(boot_r, 100 * alpha / 2)
            ci_u = np.percentile(boot_r, 100 * (1 - alpha / 2))
            return float(ci_l), float(ci_u), float(np.std(boot_r))
        except Exception:
            pass

    # Fallback: Fisher z-transform CI
    r, _ = pearsonr(x, y)
    n    = len(x)
    z    = np.arctanh(np.clip(r, -0.9999, 0.9999))
    se   = 1.0 / np.sqrt(n - 3)
    return (float(np.tanh(z - 1.96 * se)),
            float(np.tanh(z + 1.96 * se)),
            float(se))


# --- Helper: permutation test ---------------------------------------------

def permutation_pval(x, y, n_perm=N_PERMUTE):
    """
    Two-sided permutation p-value for Pearson r.
    Randomly shuffles y (the ice scalar) to break any time structure,
    building the null distribution of r under no relationship.
    """
    if len(x) < 8:
        return np.nan
    r_obs = pearsonr(x, y)[0]
    rng   = np.random.default_rng(42)
    null  = np.array([
        pearsonr(x, rng.permutation(y))[0]
        for _ in range(n_perm)
    ])
    return float(np.mean(np.abs(null) >= np.abs(r_obs)))


# --- 1. Monthly cross-correlations ----------------------------------------

print("\n1. Computing monthly cross-correlations...")
cross_corr_records = []

for sec_col, sec_label in SECTORS.items():
    ice = annual_dt[annual_dt["sector"] == sec_col]

    for apac_var, var_type in APAC_VARS.items():
        ice_sub = ice[["Year", apac_var]].dropna()

        for idx_col in INDEX_COLS_MONTHLY:
            for month in range(1, 13):
                # Index value for this month, each year
                idx_m = monthly_idx[monthly_idx["month"] == month][
                    ["year", idx_col]].dropna()
                idx_m = idx_m.rename(columns={"year": "Year"})

                merged = ice_sub.merge(idx_m, on="Year", how="inner").dropna()
                if len(merged) < 8:
                    continue

                x = merged[idx_col].values.astype(float)
                y = merged[apac_var].values.astype(float)

                r, p_raw     = pearsonr(x, y)
                rho, p_spear = spearmanr(x, y)
                ci_l, ci_u, boot_std = bootstrap_ci(x, y)
                p_perm       = permutation_pval(x, y)

                cross_corr_records.append({
                    "sector"      : sec_col,
                    "sector_label": sec_label,
                    "apac_var"    : apac_var,
                    "var_type"    : var_type,
                    "index"       : idx_col,
                    "month"       : month,
                    "month_name"  : MONTH_NAMES[month - 1],
                    "n"           : len(merged),
                    "pearson_r"   : round(r, 4),
                    "pearson_p"   : round(p_raw, 4),
                    "spearman_r"  : round(rho, 4),
                    "spearman_p"  : round(p_spear, 4),
                    "ci_lower"    : round(ci_l, 4) if not np.isnan(ci_l) else np.nan,
                    "ci_upper"    : round(ci_u, 4) if not np.isnan(ci_u) else np.nan,
                    "boot_std"    : round(boot_std, 4) if not np.isnan(boot_std) else np.nan,
                    "p_permute"   : round(p_perm, 4),
                    "bootstrap_method": "stationary_block" if HAS_ARCH else "fisher_z",
                })

cross_df = pd.DataFrame(cross_corr_records)
print(f"  {len(cross_df)} monthly cross-correlation pairs")


# --- 2. Atmospheric autocorrelation structure -----------------------------

print("\n2. Computing atmospheric autocorrelation structure...")
autocorr_records = []

for idx_col in INDEX_COLS_MONTHLY:
    for month in range(1, 13):
        series = (monthly_idx[monthly_idx["month"] == month]
                  .sort_values("year")[idx_col]
                  .dropna().values.astype(float))

        if len(series) < 10:
            continue

        for lag in range(1, 13):
            if lag >= len(series):
                break
            r_ac = np.corrcoef(series[:-lag], series[lag:])[0, 1]
            autocorr_records.append({
                "index"     : idx_col,
                "month"     : month,
                "month_name": MONTH_NAMES[month - 1],
                "lag"       : lag,
                "autocorr_r": round(r_ac, 4),
            })

autocorr_df = pd.DataFrame(autocorr_records)
print(f"  {len(autocorr_df)} autocorrelation values")


# --- 3. Partial correlations at peak lags ---------------------------------
# For each key pair, find the month with the strongest cross-correlation,
# then compute the partial correlation controlling for the index's own
# lag-1 autocorrelation in that month.

print("\n3. Computing partial correlations at peak lags...")
partial_records = []

for sec_col, apac_var, idx_base in KEY_PAIRS:
    # Find which index column matches (SAM, ZW3R, ASL, Nino34)
    matching = [c for c in INDEX_COLS_MONTHLY if c.startswith(idx_base)]
    if not matching:
        # Try annual index from idx table
        annual_col = f"{idx_base}_annual"
        if annual_col not in idx.columns:
            continue
        matching = [annual_col]
    idx_col = matching[0]

    if idx_col not in INDEX_COLS_MONTHLY:
        continue

    ice = annual_dt[annual_dt["sector"] == sec_col][["Year", apac_var]].dropna()

    # Find peak month for this pair
    pair_rows = cross_df[
        (cross_df["sector"] == sec_col) &
        (cross_df["apac_var"] == apac_var) &
        (cross_df["index"] == idx_col)
    ]
    if len(pair_rows) == 0:
        continue

    peak_month = int(pair_rows.loc[pair_rows["pearson_r"].abs().idxmax(), "month"])

    # Get index at peak month and at peak month - 1 (for partial)
    idx_peak = (monthly_idx[monthly_idx["month"] == peak_month]
                [["year", idx_col]].dropna()
                .rename(columns={"year":"Year", idx_col:"idx_peak"}))

    prev_month = ((peak_month - 2) % 12) + 1
    idx_prev = (monthly_idx[monthly_idx["month"] == prev_month]
                [["year", idx_col]].dropna()
                .rename(columns={"year":"Year", idx_col:"idx_prev"}))

    merged = (ice
              .merge(idx_peak, on="Year")
              .merge(idx_prev, on="Year")
              .dropna())

    if len(merged) < 8:
        continue

    y  = merged[apac_var].values.astype(float)
    x1 = merged["idx_peak"].values.astype(float)
    x2 = merged["idx_prev"].values.astype(float)

    # Simple partial r: residualise y and x1 on x2
    def residualise(a, b):
        s, i, *_ = stats.linregress(b, a)
        return a - (s * b + i)

    y_resid  = residualise(y,  x2)
    x1_resid = residualise(x1, x2)
    r_partial, p_partial = pearsonr(x1_resid, y_resid)
    r_raw,     p_raw     = pearsonr(x1, y)

    partial_records.append({
        "sector"       : sec_col,
        "apac_var"     : apac_var,
        "index"        : idx_col,
        "peak_month"   : peak_month,
        "peak_month_name": MONTH_NAMES[peak_month - 1],
        "control_month": prev_month,
        "r_raw"        : round(r_raw,     4),
        "p_raw"        : round(p_raw,     4),
        "r_partial"    : round(r_partial, 4),
        "p_partial"    : round(p_partial, 4),
        "n"            : len(merged),
    })

partial_df = pd.DataFrame(partial_records)
print(f"  {len(partial_df)} partial correlation pairs")


# --- 4. Leave-one-out sensitivity -----------------------------------------

print("\n4. Computing leave-one-out sensitivity for key pairs...")
loo_records = []

for sec_col, apac_var, idx_base in KEY_PAIRS:
    matching = [c for c in INDEX_COLS_MONTHLY if c.startswith(idx_base)]
    if not matching:
        continue
    idx_col = matching[0]

    # Use the annual mean of the index (most comparable to existing results)
    idx_ann = (monthly_idx.groupby("year")[idx_col]
               .mean().reset_index()
               .rename(columns={"year":"Year"}))

    ice = annual_dt[annual_dt["sector"] == sec_col][["Year", apac_var]].dropna()
    merged = ice.merge(idx_ann, on="Year").dropna()

    if len(merged) < 8:
        continue

    x    = merged[idx_col].values.astype(float)
    y    = merged[apac_var].values.astype(float)
    yrs  = merged["Year"].values

    r_full, _ = pearsonr(x, y)

    for i, yr in enumerate(yrs):
        mask  = np.arange(len(yrs)) != i
        r_loo, _ = pearsonr(x[mask], y[mask])
        loo_records.append({
            "sector"      : sec_col,
            "apac_var"    : apac_var,
            "index"       : idx_col,
            "year_left_out": int(yr),
            "r_full"      : round(r_full, 4),
            "r_loo"       : round(r_loo,  4),
            "r_change"    : round(r_loo - r_full, 4),
        })

loo_df = pd.DataFrame(loo_records)
print(f"  {len(loo_df)} leave-one-out records")

# Flag high-leverage years — those where removing them changes r by > 0.05
loo_df["high_leverage"] = loo_df["r_change"].abs() > 0.05
n_lev = loo_df["high_leverage"].sum()
print(f"  High-leverage years (|Δr| > 0.05): {n_lev}")
print(loo_df[loo_df["high_leverage"]][
    ["sector","apac_var","index","year_left_out","r_full","r_loo","r_change"]
].to_string(index=False))


# --- Save all outputs -----------------------------------------------------

outputs = {
    "monthly_cross_correlations.csv" : cross_df,
    "atmospheric_autocorrelations.csv": autocorr_df,
    "partial_correlations.csv"        : partial_df,
    "loo_sensitivity.csv"             : loo_df,
}

print("\nSaving outputs...")
for fname, df in outputs.items():
    path = os.path.join(DATA_DIR, fname)
    df.to_csv(path, index=False)
    print(f"  {path}  ({len(df)} rows)")

print("\nDone.")