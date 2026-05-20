"""
compute_monthly_correlations_both.py
=====================================
Two types of monthly correlations:

PART 1 — LAG CORRELATION:
  Response:  annual APAC amplitude_anom / max_doy_anom (one value per year)
  Predictor: monthly index value in month M
  For each calendar month M, correlate index(M,year) with annual scalar(year)
  n = 44 per month
  Answers: "Which month's atmosphere best predicts the annual ice cycle?"

PART 2 — CONTEMPORANEOUS MONTHLY CORRELATION:
  Response:  monthly_amp_anom from monthly_params.csv (one value per year per month)
  Predictor: monthly index value in same month and year
  n = 44 per month (one year per data point)
  Answers: "Does the monthly atmosphere co-vary with monthly ice amplitude?"

Outputs:
  lag_correlations.csv            — Part 1 results
  contemporaneous_correlations.csv — Part 2 results
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from scipy import stats
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR  = "/user/geog/falejandraperez/sea-ice-phase/scripts/R/Ch3/data"
INDEX_DIR = "/user/geog/falejandraperez/sea-ice-phase/data/indices"

ANNUAL_CSV  = os.path.join(DATA_DIR, "annual_params.csv")
MONTHLY_CSV = os.path.join(DATA_DIR, "monthly_params.csv")

YEAR_MIN = 1979
YEAR_MAX = 2023

SECTORS = {
    "SIE_Weddell"                : "Weddell",
    "SIE_Amundsen_Bellingshausen": "ABS",
    "SIE_Ross"                   : "Ross",
    "SIE_East_Antarctica"        : "East Antarctica",
    "SIE_King_Haakon"            : "King Haakon",
}

month_map = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,
             "Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}
MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]

# ── Load ice data ─────────────────────────────────────────────────────────────
print("Loading ice data...")
annual  = pd.read_csv(ANNUAL_CSV)
annual  = annual[annual["Year"].between(YEAR_MIN, YEAR_MAX)]
monthly = pd.read_csv(MONTHLY_CSV)
monthly = monthly[monthly["Year"].between(YEAR_MIN, YEAR_MAX)]
print(f"  Annual:  {len(annual)} rows")
print(f"  Monthly: {len(monthly)} rows")

# ── Load monthly indices ──────────────────────────────────────────────────────
print("\nLoading monthly indices...")

# SAM
sam_raw = pd.read_csv(
    os.path.join(INDEX_DIR, "marshall_sam_monthly.txt"),
    delim_whitespace=True, header=0,
    names=["year","Jan","Feb","Mar","Apr","May","Jun",
           "Jul","Aug","Sep","Oct","Nov","Dec"])
sam_long = sam_raw.melt(id_vars="year", var_name="month_str", value_name="SAM")
sam_long["month"] = sam_long["month_str"].map(month_map)
sam_long = sam_long[["year","month","SAM"]].dropna()
sam_long.columns = ["Year","Month","SAM"]

# Niño3.4
nino_raw = pd.read_csv(
    os.path.join(INDEX_DIR, "nina34.data"),
    sep=r'\s+', skiprows=1, header=None,
    names=["year","Jan","Feb","Mar","Apr","May","Jun",
           "Jul","Aug","Sep","Oct","Nov","Dec"],
    na_values=["-99.99", -99.99], engine="python")
nino_raw = nino_raw[pd.to_numeric(nino_raw["year"], errors="coerce").between(1900,2100)]
nino_raw["year"] = nino_raw["year"].astype(float).astype(int)
nino_long = nino_raw.melt(id_vars="year", var_name="month_str", value_name="Nino34")
nino_long["month"] = nino_long["month_str"].map(month_map)
nino_long = nino_long[["year","month","Nino34"]].dropna()
nino_long.columns = ["Year","Month","Nino34"]

# ASL
asl_raw = pd.read_csv(os.path.join(INDEX_DIR, "asli_era5_v3-latest.csv"), comment="#")
asl_raw["time"]  = pd.to_datetime(asl_raw["time"])
asl_raw["Year"]  = asl_raw["time"].dt.year
asl_raw["Month"] = asl_raw["time"].dt.month
asl_long = asl_raw[["Year","Month","RelCenPres"]].rename(columns={"RelCenPres":"ASL"})

# ZW3R
zw3_raw = pd.read_csv(os.path.join(INDEX_DIR, "ZW3_raphael_monthly.csv"))
zw3_long = zw3_raw[["year","month","ZW3_index"]].copy()
zw3_long.columns = ["Year","Month","ZW3R"]

# Merge all indices
print("  Merging indices...")
idx_monthly = (sam_long
               .merge(nino_long, on=["Year","Month"], how="outer")
               .merge(asl_long,  on=["Year","Month"], how="outer")
               .merge(zw3_long,  on=["Year","Month"], how="outer"))
idx_monthly = idx_monthly[idx_monthly["Year"].between(YEAR_MIN, YEAR_MAX)]
print(f"  Monthly index table: {idx_monthly.shape}")

INDICES = ["SAM","Nino34","ASL","ZW3R"]

# ── Helper: detrend ───────────────────────────────────────────────────────────
def detrend(years, values):
    mask = ~np.isnan(values)
    if mask.sum() < 5: return values
    s, i, *_ = stats.linregress(years[mask].astype(float), values[mask])
    return values - (s * years.astype(float) + i)

# ── PART 1: Lag correlations ──────────────────────────────────────────────────
print("\n=== PART 1: Lag correlations (annual scalar ~ monthly index) ===")

lag_results = []

for sec_col, sec_label in SECTORS.items():
    sec = annual[annual["sector"] == sec_col].copy().sort_values("Year")

    for ice_var, var_label in [("amplitude_anom","amplitude"), ("max_doy_anom","phase")]:
        if ice_var not in sec.columns: continue
        y_annual = detrend(sec["Year"].values, sec[ice_var].values.astype(float))
        y_df = pd.DataFrame({"Year": sec["Year"].values, "y": y_annual})

        for idx_name in INDICES:
            for month in range(1, 13):
                idx_sub = idx_monthly[idx_monthly["Month"] == month][["Year", idx_name]].dropna()
                merged  = y_df.merge(idx_sub, on="Year").dropna()
                if len(merged) < 10: continue

                x = detrend(merged["Year"].values, merged[idx_name].values.astype(float))
                y = merged["y"].values

                r, p = pearsonr(x, y)
                lag_results.append({
                    "sector"      : sec_label,
                    "var_type"    : var_label,
                    "index"       : idx_name,
                    "month"       : month,
                    "month_name"  : MONTH_NAMES[month-1],
                    "n"           : len(merged),
                    "pearson_r"   : round(r, 4),
                    "pearson_p"   : round(p, 4),
                    "type"        : "lag"
                })

lag_df = pd.DataFrame(lag_results)

# FDR per sector × variable × index group
lag_fdr = []
for (sec, var, idx), grp in lag_df.groupby(["sector","var_type","index"]):
    grp = grp.copy()
    _, p_adj, _, _ = multipletests(grp["pearson_p"].values, alpha=0.05, method="fdr_bh")
    grp["p_fdr"] = np.round(p_adj, 4)
    grp["sig"]   = grp["p_fdr"] < 0.05
    lag_fdr.append(grp)
lag_df = pd.concat(lag_fdr)

out1 = os.path.join(DATA_DIR, "lag_correlations.csv")
lag_df.to_csv(out1, index=False)
print(f"  Saved → {out1} ({len(lag_df)} rows)")

# ── PART 2: Contemporaneous monthly correlations ──────────────────────────────
print("\n=== PART 2: Contemporaneous monthly correlations ===")

cont_results = []

for sec_col, sec_label in SECTORS.items():
    sec_monthly = monthly[monthly["sector"] == sec_col].copy()

    for ice_var, var_label in [("monthly_amp_anom","amplitude"),
                                ("monthly_mean_anom","mean_sie")]:
        if ice_var not in sec_monthly.columns: continue

        for idx_name in INDICES:
            for month in range(1, 13):
                # Get ice values for this month across all years
                ice_sub = sec_monthly[sec_monthly["Month"] == month][
                    ["Year", ice_var]].dropna()

                # Get index values for this month
                idx_sub = idx_monthly[idx_monthly["Month"] == month][
                    ["Year", idx_name]].dropna()

                merged = ice_sub.merge(idx_sub, on="Year").dropna()
                if len(merged) < 10: continue

                x = detrend(merged["Year"].values,
                            merged[idx_name].values.astype(float))
                y = detrend(merged["Year"].values,
                            merged[ice_var].values.astype(float))

                r, p = pearsonr(x, y)
                cont_results.append({
                    "sector"      : sec_label,
                    "var_type"    : var_label,
                    "index"       : idx_name,
                    "month"       : month,
                    "month_name"  : MONTH_NAMES[month-1],
                    "n"           : len(merged),
                    "pearson_r"   : round(r, 4),
                    "pearson_p"   : round(p, 4),
                    "type"        : "contemporaneous"
                })

cont_df = pd.DataFrame(cont_results)

# FDR
cont_fdr = []
for (sec, var, idx), grp in cont_df.groupby(["sector","var_type","index"]):
    grp = grp.copy()
    _, p_adj, _, _ = multipletests(grp["pearson_p"].values, alpha=0.05, method="fdr_bh")
    grp["p_fdr"] = np.round(p_adj, 4)
    grp["sig"]   = grp["p_fdr"] < 0.05
    cont_fdr.append(grp)
cont_df = pd.concat(cont_fdr)

out2 = os.path.join(DATA_DIR, "contemporaneous_correlations.csv")
cont_df.to_csv(out2, index=False)
print(f"  Saved → {out2} ({len(cont_df)} rows)")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n=== Summary ===")
print(f"Lag correlations:            {len(lag_df)} pairs, {lag_df['sig'].sum()} FDR significant")
print(f"Contemporaneous correlations: {len(cont_df)} pairs, {cont_df['sig'].sum()} FDR significant")

print("\nTop 10 lag correlations by |r|:")
print(lag_df.nlargest(10, "pearson_r", keep="all")
      [["sector","var_type","index","month_name","pearson_r","pearson_p","p_fdr","sig"]]
      .to_string(index=False))

print("\nTop 10 contemporaneous correlations by |r|:")
top_cont = cont_df.reindex(cont_df["pearson_r"].abs().sort_values(ascending=False).index)
print(top_cont.head(10)
      [["sector","var_type","index","month_name","pearson_r","pearson_p","p_fdr","sig"]]
      .to_string(index=False))

print("\nDone.")