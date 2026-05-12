"""
Computes correlations between APAC phase/amplitude anomalies and atmospheric
indices across all sectors and seasons.

The FDR correction is applied separately within each variable type (phase /
amplitude) AND within each index — so we're not over-correcting across
physically unrelated tests. For example, SAM correlations with phase are
corrected separately from ZW3 correlations with amplitude.

Indices:
    SAM     — Marshall (2003) observational index, monthly
    ZW3R    — Raphael annual index
    ASL     — Amundsen Sea Low relative central pressure, monthly ERA5 v3
    Nino3.4 — CPC ERSSTv5 Nino 3.4 mean, monthly

Seasons: DJF, MAM, JJA, SON, annual

Outputs:
    correlations_output.csv     — one row per (sector, variable, index, season)
    master_index_detrended.csv  — detrended seasonal index table, wide format
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

INDEX_DIR  = "/user/geog/falejandraperez/sea-ice-phase/data/indices"
DATA_DIR   = "/user/geog/falejandraperez/sea-ice-phase/scripts/R/chapter3/data"
OUTPUT_DIR = DATA_DIR

ANNUAL_CSV = os.path.join(DATA_DIR, "annual_params.csv")
YEAR_MIN   = 1979
YEAR_MAX   = 2023

SECTORS = {
    "SIE_Weddell"                : "Weddell",
    "SIE_Amundsen_Bellingshausen": "ABS",
    "SIE_Ross"                   : "Ross",
    "SIE_East_Antarctica"        : "East Antarctica",
    "SIE_King_Haakon"            : "King Haakon",
}

APAC_VARS = {
    "amplitude_anom": "amplitude",
    "max_doy_anom"  : "phase",
}

month_map = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,
             "Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}


# --- Load APAC annual parameters ------------------------------------------

print("Loading APAC annual parameters...")
annual = pd.read_csv(ANNUAL_CSV)
annual = annual[annual["Year"].between(YEAR_MIN, YEAR_MAX)]

for col in list(APAC_VARS.keys()):
    if col in annual.columns:
        annual[col] = pd.to_numeric(annual[col], errors="coerce")

print(f"  {len(annual)} rows | sectors: {annual['sector'].nunique()}")


# --- Helpers --------------------------------------------------------------

def compute_seasonal_means(df_monthly, value_col, year_col="year",
                            month_col="month"):
    df = df_monthly.copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    annual_mean = (df.groupby(year_col)[value_col]
                   .mean(numeric_only=True).reset_index()
                   .rename(columns={value_col: "annual"}))

    season_map = {12:"DJF", 1:"DJF", 2:"DJF",
                  3:"MAM", 4:"MAM", 5:"MAM",
                  6:"JJA", 7:"JJA", 8:"JJA",
                  9:"SON", 10:"SON", 11:"SON"}
    df["season"] = df[month_col].map(season_map)
    df["season_year"] = df[year_col]
    df.loc[df[month_col] == 12, "season_year"] = (
        df.loc[df[month_col] == 12, year_col] + 1)

    seasonal = (df.groupby(["season_year","season"])[value_col]
                .mean(numeric_only=True).unstack("season")
                .reset_index()
                .rename(columns={"season_year": year_col}))

    return annual_mean.merge(seasonal, on=year_col, how="outer")


def detrend_series(years, values):
    mask = ~np.isnan(values)
    if mask.sum() < 5:
        return values
    s, i, *_ = stats.linregress(years[mask].astype(float), values[mask])
    return values - (s * years.astype(float) + i)


def n_eff(x, y):
    n = len(x)
    if n < 6:
        return n
    r1x = np.corrcoef(x[:-1], x[1:])[0,1] if len(x) > 2 else 0
    r1y = np.corrcoef(y[:-1], y[1:])[0,1] if len(y) > 2 else 0
    denom = 1 + r1x * r1y
    if denom <= 0:
        return n
    return max(3, int(n * (1 - r1x * r1y) / denom))


def pearson_with_neff(x, y):
    r, _ = pearsonr(x, y)
    ne   = n_eff(x, y)
    t    = r * np.sqrt((ne - 2) / (1 - r**2 + 1e-12))
    p    = 2 * stats.t.sf(np.abs(t), df=ne - 2)
    return float(r), float(p), int(ne)


# --- Load indices ---------------------------------------------------------

print("Loading SAM index...")
sam_raw = pd.read_csv(
    os.path.join(INDEX_DIR, "marshall_sam_monthly.txt"),
    delim_whitespace=True, header=0,
    names=["year","Jan","Feb","Mar","Apr","May","Jun",
           "Jul","Aug","Sep","Oct","Nov","Dec"]
)
sam_long = sam_raw.melt(id_vars="year", var_name="month_str", value_name="SAM")
sam_long["month"] = sam_long["month_str"].map(month_map)
sam_long = sam_long.dropna(subset=["SAM"])
sam_seas = compute_seasonal_means(sam_long, "SAM")
sam_seas = sam_seas.rename(columns={
    "annual":"SAM_annual","DJF":"SAM_DJF",
    "MAM":"SAM_MAM","JJA":"SAM_JJA","SON":"SAM_SON"})
print(f"  SAM: {sam_seas['year'].min()}–{sam_seas['year'].max()}")

print("Loading ZW3 Raphael monthly index...")
zw3_monthly_r = pd.read_csv(os.path.join(INDEX_DIR, "ZW3_raphael_monthly.csv"))
zw3_monthly_r = zw3_monthly_r[["year","month","ZW3_index"]].rename(
    columns={"ZW3_index":"ZW3R"})
zw3_monthly_r[["year","month","ZW3R"]] = zw3_monthly_r[
    ["year","month","ZW3R"]].apply(pd.to_numeric, errors="coerce")

# Compute seasonal means from monthly — consistent with SAM and ASL
zw3 = compute_seasonal_means(zw3_monthly_r, "ZW3R")
zw3 = zw3.rename(columns={
    "annual":"ZW3R_annual","DJF":"ZW3R_DJF",
    "MAM":"ZW3R_MAM","JJA":"ZW3R_JJA","SON":"ZW3R_SON"})
print(f"  ZW3R: {zw3['year'].min():.0f}–{zw3['year'].max():.0f}")

# Goyal ZW3 monthly — supplementary, captures magnitude and phase separately
print("Loading ZW3 Goyal monthly index...")
zw3_goyal = pd.read_csv(os.path.join(INDEX_DIR, "ZW3_goyal_monthly.csv"))
zw3_goyal = zw3_goyal.rename(columns={"ZW3_magnitude": "ZW3G"})
zw3_goyal = zw3_goyal[["year","month","ZW3G"]].dropna()

zw3g_seas = compute_seasonal_means(zw3_goyal, "ZW3G")
zw3g_seas = zw3g_seas.rename(columns={
    "annual":"ZW3G_annual","DJF":"ZW3G_DJF",
    "MAM":"ZW3G_MAM","JJA":"ZW3G_JJA","SON":"ZW3G_SON"})
print(f"  ZW3G: {zw3g_seas['year'].min():.0f}–{zw3g_seas['year'].max():.0f}")

print("Loading ASL index...")
asl_raw = pd.read_csv(
    os.path.join(INDEX_DIR, "asli_era5_v3-latest.csv"), comment="#")
asl_raw["time"]  = pd.to_datetime(asl_raw["time"])
asl_raw["year"]  = asl_raw["time"].dt.year
asl_raw["month"] = asl_raw["time"].dt.month
asl_raw = asl_raw.rename(columns={"RelCenPres":"ASL"})
asl_seas = compute_seasonal_means(asl_raw, "ASL")
asl_seas = asl_seas.rename(columns={
    "annual":"ASL_annual","DJF":"ASL_DJF",
    "MAM":"ASL_MAM","JJA":"ASL_JJA","SON":"ASL_SON"})
print(f"  ASL: {asl_seas['year'].min():.0f}–{asl_seas['year'].max():.0f}")

print("Loading Niño3.4 index...")
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

nino_long = nino_raw.melt(id_vars="year", var_name="month_str", value_name="Nino34")
nino_long["month"] = nino_long["month_str"].map(month_map)
nino_long = nino_long.dropna(subset=["Nino34"]).drop(columns="month_str")
nino_seas = compute_seasonal_means(nino_long, "Nino34")
nino_seas = nino_seas.rename(columns={
    "annual":"Nino34_annual","DJF":"Nino34_DJF",
    "MAM":"Nino34_MAM","JJA":"Nino34_JJA","SON":"Nino34_SON"})
print(f"  Nino34: {nino_seas['year'].min():.0f}–{nino_seas['year'].max():.0f}")


# --- Merge and detrend indices --------------------------------------------

print("\nMerging indices...")
idx = (sam_seas
       .merge(zw3,        on="year", how="outer")
       .merge(zw3g_seas,  on="year", how="outer")
       .merge(asl_seas,   on="year", how="outer")
       .merge(nino_seas,  on="year", how="outer"))

idx = idx[idx["year"].between(YEAR_MIN, YEAR_MAX)].sort_values("year")

for col in [c for c in idx.columns if c != "year"]:
    idx[col] = detrend_series(idx["year"].values, idx[col].values.astype(float))

idx = idx.rename(columns={"year": "Year"})
print(f"  Master index table: {idx.shape} | {idx['Year'].min()}–{idx['Year'].max()}")

idx_out = os.path.join(OUTPUT_DIR, "master_index_detrended.csv")
idx.to_csv(idx_out, index=False)
print(f"  Saved: {idx_out}")


# --- Detrend APAC variables -----------------------------------------------

annual_dt = []
for sec_col in SECTORS:
    sec = annual[annual["sector"] == sec_col].copy().sort_values("Year")
    x   = sec["Year"].values.astype(float)
    for var in APAC_VARS:
        if var in sec.columns:
            sec[var] = detrend_series(x, sec[var].values.astype(float))
    annual_dt.append(sec)
annual_dt = pd.concat(annual_dt)


# --- Compute correlations -------------------------------------------------
# FDR applied within each (index × variable_type) group — so SAM~amplitude
# is corrected separately from ZW3~amplitude. This avoids over-correction
# from pooling physically unrelated tests together.

print("\nComputing correlations...")

INDEX_COLS = [c for c in idx.columns if c != "Year"]
results = []

for sec_col, sec_label in SECTORS.items():
    sec_data = annual_dt[annual_dt["sector"] == sec_col].copy()

    for apac_var, var_type in APAC_VARS.items():
        apac = sec_data[["Year", apac_var]].dropna()

        for idx_col in INDEX_COLS:
            idx_sub = idx[["Year", idx_col]].dropna()
            merged  = apac.merge(idx_sub, on="Year", how="inner").dropna()

            if len(merged) < 8:
                continue

            x = merged[idx_col].values.astype(float)
            y = merged[apac_var].values.astype(float)

            r, p_neff, ne = pearson_with_neff(x, y)
            rho, p_spear  = spearmanr(x, y)

            parts    = idx_col.rsplit("_", 1)
            idx_name = parts[0]
            season   = parts[1] if len(parts) == 2 else "annual"

            results.append({
                "sector"      : sec_col,
                "sector_label": sec_label,
                "variable"    : apac_var,
                "var_type"    : var_type,
                "index"       : idx_name,
                "season"      : season,
                "n"           : len(merged),
                "n_eff"       : ne,
                "pearson_r"   : round(r,   4),
                "pearson_p"   : round(p_neff, 4),
                "spearman_r"  : round(rho, 4),
                "spearman_p"  : round(float(p_spear), 4),
            })

results_df = pd.DataFrame(results)
print(f"  {len(results_df)} correlation pairs computed")


# --- FDR correction -------------------------------------------------------
# Applied within each (var_type × index) group — physically sensible grouping
# that avoids over-correcting across unrelated tests.

print("Applying FDR correction (Benjamini-Hochberg)...")

fdr_results = []
for (var_type, index), group in results_df.groupby(["var_type", "index"]):
    group = group.copy()

    reject_p, p_adj_p, _, _ = multipletests(
        group["pearson_p"].values, alpha=0.05, method="fdr_bh")
    group["pearson_p_fdr"] = np.round(p_adj_p, 4)
    group["pearson_sig"]   = reject_p.astype(bool)

    reject_s, p_adj_s, _, _ = multipletests(
        group["spearman_p"].values, alpha=0.05, method="fdr_bh")
    group["spearman_p_fdr"] = np.round(p_adj_s, 4)
    group["spearman_sig"]   = reject_s.astype(bool)

    fdr_results.append(group)

results_df = pd.concat(fdr_results).sort_values(
    ["sector","var_type","index","season"]).reset_index(drop=True)

# Significance stars
def sig_star(row):
    if row["pearson_p_fdr"] < 0.05: return "**"
    if row["pearson_p"]     < 0.05: return "*"
    if row["pearson_p"]     < 0.10: return "."
    return ""

results_df["sig"] = results_df.apply(sig_star, axis=1)

n_sig = int(results_df["pearson_sig"].sum())
print(f"  {n_sig} significant after FDR (across index×variable groups)")


# --- Save -----------------------------------------------------------------

out_path = os.path.join(OUTPUT_DIR, "correlations_output.csv")
results_df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")

print("\nTop 10 by |Pearson r|:")
top = (results_df
       .copy()
       .assign(abs_r=lambda df: df["pearson_r"].abs())
       .sort_values("abs_r", ascending=False)
       .head(10)
       [["sector_label","var_type","index","season",
         "pearson_r","pearson_p","pearson_p_fdr","n_eff","sig"]])
print(top.to_string(index=False))

print("\nFDR significant results:")
sig = (results_df[results_df["pearson_sig"]]
       .sort_values("pearson_r", key=abs, ascending=False)
       [["sector_label","var_type","index","season",
         "pearson_r","pearson_p_fdr","n_eff","sig"]])
print(sig.to_string(index=False) if len(sig) > 0 else "  None")