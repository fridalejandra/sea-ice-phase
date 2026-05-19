"""
compute_loo_index.py
====================
Leave-one-index-out regression analysis.

For each sector × variable (phase / amplitude), fits simple OLS regressions
using each atmospheric index as the sole predictor, plus a multi-index
model (ALL) as the reference. Records RMSE, R², and year-by-year absolute
residuals for each scenario.

Models fit:
    ALL      — all available indices as predictors (multiple regression)
    noSAM    — all indices except SAM
    noZW3    — all indices except ZW3R
    noASL    — all indices except ASL
    noNino34 — all indices except Nino34
    SAM_only, ZW3_only, ASL_only, Nino34_only — single-predictor models

Outputs (saved to DATA_DIR):
    loo_index_skill.csv     — RMSE, R², adj-R² per sector/variable/scenario
    loo_index_residuals.csv — year-by-year absolute residuals per scenario

Needs:
    annual_params.csv           — APAC annual scalars
    master_index_detrended.csv  — seasonal/annual index means (from
                                  compute_atmospheric_correlations.py)
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR  = "/user/geog/falejandraperez/sea-ice-phase/scripts/R/Ch3/data"
ANNUAL_CSV = os.path.join(DATA_DIR, "annual_params.csv")
INDEX_CSV  = os.path.join(DATA_DIR, "master_index_detrended.csv")

YEAR_MIN = 1979
YEAR_MAX = 2023

# ── Sector / variable definitions ─────────────────────────────────────────────
SECTORS = {
    "SIE_Weddell"                : "Weddell",
    "SIE_Amundsen_Bellingshausen": "ABS",
    "SIE_Ross"                   : "Ross",
    "SIE_East_Antarctica"        : "EA",
    "SIE_King_Haakon"            : "King Haakon",
}

APAC_VARS = {
    "amplitude_anom": "amplitude",
    "max_doy_anom"  : "phase",
}

# Annual index columns in master_index_detrended.csv
# Adjust these names to match your actual column headers
INDEX_COLS = ["SAM_annual", "ZW3R_annual", "ASL_annual", "Nino34_annual"]
INDEX_LABELS = {
    "SAM_annual"   : "SAM",
    "ZW3R_annual"  : "ZW3",
    "ASL_annual"   : "ASL",
    "Nino34_annual": "Niño3.4",
}

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
annual = pd.read_csv(ANNUAL_CSV)
annual = annual[annual["Year"].between(YEAR_MIN, YEAR_MAX)]
for col in list(APAC_VARS.keys()):
    if col in annual.columns:
        annual[col] = pd.to_numeric(annual[col], errors="coerce")

idx = pd.read_csv(INDEX_CSV)
idx = idx[idx["Year"].between(YEAR_MIN, YEAR_MAX)]

# Confirm which index columns are actually present
available = [c for c in INDEX_COLS if c in idx.columns]
missing   = [c for c in INDEX_COLS if c not in idx.columns]
if missing:
    print(f"  WARNING: index columns not found and will be skipped: {missing}")
INDEX_COLS = available
print(f"  Using index columns: {INDEX_COLS}")

# ── Detrend APAC scalars ──────────────────────────────────────────────────────
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

# ── Build scenarios ───────────────────────────────────────────────────────────
# Each scenario is a dict: name -> list of predictor columns to use
scenarios = {"ALL": INDEX_COLS}
for col in INDEX_COLS:
    label = INDEX_LABELS.get(col, col)
    # Leave-one-out: all except this index
    scenarios[f"no{label}"] = [c for c in INDEX_COLS if c != col]
    # Single-predictor
    scenarios[f"{label}_only"] = [col]

print(f"\nScenarios: {list(scenarios.keys())}")

# ── OLS helper ────────────────────────────────────────────────────────────────
def fit_ols(X, y):
    """
    Fit OLS, return dict with rmse, r2, adj_r2, predictions, residuals.
    X: (n, p) array, y: (n,) array.
    """
    n, p = X.shape
    reg  = LinearRegression().fit(X, y)
    yhat = reg.predict(X)
    rmse = np.sqrt(mean_squared_error(y, yhat))
    r2   = r2_score(y, yhat)
    # Adjusted R²
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else np.nan
    return {
        "rmse"    : rmse,
        "r2"      : r2,
        "adj_r2"  : adj_r2,
        "yhat"    : yhat,
        "resid"   : np.abs(y - yhat),
        "coefs"   : dict(zip(
            [f"coef_{c}" for c in scenarios.get("ALL", [])[:p]], reg.coef_
        )),
    }

# ── Main loop ─────────────────────────────────────────────────────────────────
skill_records  = []
resid_records  = []

for sec_col, sec_label in SECTORS.items():
    for apac_var, var_label in APAC_VARS.items():

        # Ice scalar for this sector/variable
        ice = (annual_dt[annual_dt["sector"] == sec_col][["Year", apac_var]]
               .dropna().sort_values("Year"))

        # Merge with all index columns
        merged = ice.merge(idx[["Year"] + INDEX_COLS], on="Year", how="inner").dropna()
        if len(merged) < 10:
            print(f"  Skipping {sec_label} {var_label}: only {len(merged)} complete rows")
            continue

        y    = merged[apac_var].values.astype(float)
        yrs  = merged["Year"].values

        for scenario_name, pred_cols in scenarios.items():
            # Skip if none of the predictor columns are available in merged
            valid_cols = [c for c in pred_cols if c in merged.columns]
            if not valid_cols:
                continue

            X = merged[valid_cols].values.astype(float)

            result = fit_ols(X, y)

            skill_records.append({
                "sector"  : sec_label,
                "variable": var_label,
                "scenario": scenario_name,
                "n"       : len(merged),
                "n_pred"  : len(valid_cols),
                "rmse"    : round(result["rmse"],   4),
                "r2"      : round(result["r2"],     4),
                "adj_r2"  : round(result["adj_r2"], 4) if not np.isnan(result["adj_r2"]) else np.nan,
            })

            for i, yr in enumerate(yrs):
                resid_records.append({
                    "sector"   : sec_label,
                    "variable" : var_label,
                    "scenario" : scenario_name,
                    "year"     : int(yr),
                    "abs_resid": round(result["resid"][i], 4),
                    "yhat"     : round(result["yhat"][i],  4),
                })

# ── Save ──────────────────────────────────────────────────────────────────────
skill_df = pd.DataFrame(skill_records)
resid_df = pd.DataFrame(resid_records)

skill_path = os.path.join(DATA_DIR, "loo_index_skill.csv")
resid_path = os.path.join(DATA_DIR, "loo_index_residuals.csv")

skill_df.to_csv(skill_path, index=False)
resid_df.to_csv(resid_path, index=False)

print(f"\nSaved → {skill_path}  ({len(skill_df)} rows)")
print(f"Saved → {resid_path}  ({len(resid_df)} rows)")

# ── Quick summary ─────────────────────────────────────────────────────────────
print("\nSkill summary (ALL model):")
all_skill = skill_df[skill_df["scenario"] == "ALL"][
    ["sector", "variable", "rmse", "r2", "adj_r2"]
].sort_values(["variable", "sector"])
print(all_skill.to_string(index=False))

print("\nDone.")