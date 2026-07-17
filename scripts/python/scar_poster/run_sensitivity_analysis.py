"""
run_sensitivity_analysis.py  (ANOMALY VERSION)

Uses SIA_anomaly / delta_SIA_anomaly (deseasonalized) instead of raw SIA,
per the Ch4 framework where X_t is defined as a deviation from the seasonal
cycle, not the raw state.

  1. Regression: delta_SIA_anomaly ~ wind_stress, split by sector, season,
     pre/post-2016 period.
  2. AR(1) persistence model on SIA_anomaly, same splits, now with:
       - an ACF check (lag-1 through lag-10) to see whether a single-lag
         AR(1) is even a reasonable representation of the memory structure
       - block-bootstrap confidence intervals on rho (3-year-equivalent
         blocks, consistent with the block bootstrap already used in Ch3)
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf

IN_CSV = "/user/geog/falejandraperez/sea-ice-phase/data/merged/analysis_table_daily_anomaly.csv"
OUT_DIR = "/user/geog/falejandraperez/sea-ice-phase/data/merged/analysis_results"
N_BOOTSTRAP = 500
BLOCK_LEN_DAYS = 365 * 3  # ~3-year blocks, matching Ch3's block bootstrap

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(IN_CSV, parse_dates=["date"])
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["period"] = np.where(df["year"] < 2016, "pre_2016", "post_2016")

season_map = {12: "DJF", 1: "DJF", 2: "DJF",
              3: "MAM", 4: "MAM", 5: "MAM",
              6: "JJA", 7: "JJA", 8: "JJA",
              9: "SON", 10: "SON", 11: "SON"}
df["season"] = df["month"].map(season_map)

sectors = df["sector"].unique()


def run_regression(sub, y_col, x_col):
    sub = sub.dropna(subset=[y_col, x_col])
    if len(sub) < 30:
        return None
    X = sm.add_constant(sub[x_col])
    y = sub[y_col]
    model = sm.OLS(y, X).fit()
    return {
        "n_obs": len(sub),
        "beta": model.params[x_col],
        "pvalue": model.pvalues[x_col],
        "r_squared": model.rsquared,
        "residual_variance": model.resid.var(),
    }


# ---------------- 1. Regression on ANOMALY delta ----------------
print("=" * 60)
print("REGRESSION: delta_SIA_anomaly ~ wind_stress")
print("=" * 60)

regression_results = []
for sector in sectors:
    for period in ["pre_2016", "post_2016"]:
        for season in ["DJF", "MAM", "JJA", "SON"]:
            sub = df[(df["sector"] == sector) & (df["period"] == period) & (df["season"] == season)]
            res = run_regression(sub, "delta_SIA_anomaly", "wind_stress")
            if res:
                row = {"sector": sector, "period": period, "season": season}
                row.update(res)
                regression_results.append(row)

regression_df = pd.DataFrame(regression_results)
regression_df.to_csv(f"{OUT_DIR}/wind_stress_regressions_anomaly.csv", index=False)
print(f"Saved {len(regression_df)} regression results")
print(regression_df.head(10))

pivot_beta = regression_df.pivot_table(index=["sector", "season"], columns="period", values="beta")
pivot_resvar = regression_df.pivot_table(index=["sector", "season"], columns="period", values="residual_variance")
pivot_r2 = regression_df.pivot_table(index=["sector", "season"], columns="period", values="r_squared")

print("\nBeta (sensitivity), pre vs post 2016:")
print(pivot_beta)
print("\nR-squared, pre vs post 2016:")
print(pivot_r2)
print("\nResidual variance, pre vs post 2016:")
print(pivot_resvar)

pivot_beta.to_csv(f"{OUT_DIR}/beta_pre_post_anomaly.csv")
pivot_resvar.to_csv(f"{OUT_DIR}/residual_variance_pre_post_anomaly.csv")
pivot_r2.to_csv(f"{OUT_DIR}/r2_pre_post_anomaly.csv")

# ---------------- 2a. ACF check - is AR(1) even reasonable? ----------------
print("\n" + "=" * 60)
print("ACF CHECK: autocorrelation of SIA_anomaly at lags 1-10 (days)")
print("=" * 60)
print("(If autocorrelation decays much slower than lag-1 alone captures,")
print(" a single-lag AR(1) is understating the memory structure.)\n")

acf_results = []
for sector in sectors:
    sub = df[df["sector"] == sector].sort_values("date")
    series = sub["SIA_anomaly"].dropna().values
    if len(series) < 50:
        continue
    acf_vals = acf(series, nlags=10, fft=True)
    print(f"{sector}: lags 1-10 = {np.round(acf_vals[1:], 3)}")
    acf_results.append({"sector": sector, **{f"lag_{i}": acf_vals[i] for i in range(1, 11)}})

pd.DataFrame(acf_results).to_csv(f"{OUT_DIR}/acf_check.csv", index=False)

# ---------------- 2b. AR(1) with block bootstrap CIs ----------------
print("\n" + "=" * 60)
print("AR(1) ON ANOMALY, WITH BLOCK-BOOTSTRAP CONFIDENCE INTERVALS")
print("=" * 60)


def fit_ar1(x_t, x_t1):
    X = sm.add_constant(x_t)
    model = sm.OLS(x_t1, X).fit()
    return model.params[1]


def block_bootstrap_rho(series, block_len, n_boot):
    """Resample contiguous blocks (with replacement) to preserve temporal
    dependence, refit AR(1) each time, return array of bootstrap rho estimates."""
    n = len(series)
    if n < block_len * 2:
        block_len = max(30, n // 4)  # shrink block length for short post-2016 series
    n_blocks_needed = int(np.ceil(n / block_len))
    rhos = []
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        starts = rng.integers(0, n - block_len, size=n_blocks_needed)
        resampled = np.concatenate([series[s:s + block_len] for s in starts])[:n]
        x_t = resampled[:-1]
        x_t1 = resampled[1:]
        if len(x_t) < 30:
            continue
        try:
            rhos.append(fit_ar1(x_t, x_t1))
        except Exception:
            continue
    return np.array(rhos)


ar1_results = []
for sector in sectors:
    for period in ["pre_2016", "post_2016"]:
        sub = df[(df["sector"] == sector) & (df["period"] == period)].sort_values("date")
        series = sub["SIA_anomaly"].dropna().values
        if len(series) < 30:
            continue

        x_t = series[:-1]
        x_t1 = series[1:]
        rho_point = fit_ar1(x_t, x_t1)

        boot_rhos = block_bootstrap_rho(series, BLOCK_LEN_DAYS, N_BOOTSTRAP)
        ci_low, ci_high = np.percentile(boot_rhos, [2.5, 97.5]) if len(boot_rhos) > 0 else (np.nan, np.nan)

        ar1_results.append({
            "sector": sector, "period": period,
            "rho": rho_point, "rho_ci_low": ci_low, "rho_ci_high": ci_high,
            "n_obs": len(series), "n_bootstrap_valid": len(boot_rhos),
        })
        print(f"  {sector} ({period}): rho = {rho_point:.4f} "
              f"[95% CI: {ci_low:.4f}, {ci_high:.4f}], n={len(series)}")

ar1_df = pd.DataFrame(ar1_results)
ar1_df.to_csv(f"{OUT_DIR}/ar1_persistence_anomaly.csv", index=False)

print("\n" + "=" * 60)
print("CHECK: do pre/post-2016 confidence intervals overlap?")
print("=" * 60)
for sector in sectors:
    rows = ar1_df[ar1_df["sector"] == sector]
    if len(rows) < 2:
        continue
    pre = rows[rows["period"] == "pre_2016"].iloc[0]
    post = rows[rows["period"] == "post_2016"].iloc[0]
    overlap = not (post["rho_ci_low"] > pre["rho_ci_high"] or post["rho_ci_high"] < pre["rho_ci_low"])
    print(f"  {sector}: pre=[{pre['rho_ci_low']:.4f},{pre['rho_ci_high']:.4f}] "
          f"post=[{post['rho_ci_low']:.4f},{post['rho_ci_high']:.4f}] "
          f"-> {'OVERLAP (not distinguishable)' if overlap else 'NO OVERLAP (likely real difference)'}")

print(f"\n\nAll results saved to: {OUT_DIR}/")