"""
run_sensitivity_analysis.py  (WIND STRESS ONLY - DLWR removed)

Core analysis for the poster:
  1. Regression: delta_SIA ~ wind_stress, split by sector, season, and
     pre/post-2016 period. beta = sensitivity, residual variance = the
     "buffering" diagnostic.
  2. AR(1) persistence model: X_t+1 = rho * X_t + noise, on raw SIA,
     same splits - the "statistical model" from your notebook.
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

IN_CSV = "/user/geog/falejandraperez/sea-ice-phase/data/merged/analysis_table_daily.csv"
OUT_DIR = "/user/geog/falejandraperez/sea-ice-phase/data/merged/analysis_results"

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
    """OLS regression of y_col on x_col. Returns dict of key stats + residuals."""
    sub = sub.dropna(subset=[y_col, x_col])
    if len(sub) < 30:
        return None, None
    X = sm.add_constant(sub[x_col])
    y = sub[y_col]
    model = sm.OLS(y, X).fit()
    result = {
        "n_obs": len(sub),
        "beta": model.params[x_col],
        "pvalue": model.pvalues[x_col],
        "r_squared": model.rsquared,
        "residual_variance": model.resid.var(),
    }
    return result, model.resid


# ---------------- 1. Univariate regression: delta_SIA ~ wind_stress ----------------
print("=" * 60)
print("REGRESSION: delta_SIA ~ wind_stress")
print("=" * 60)

regression_results = []
for sector in sectors:
    for period in ["pre_2016", "post_2016"]:
        for season in ["DJF", "MAM", "JJA", "SON"]:
            sub = df[(df["sector"] == sector) & (df["period"] == period) & (df["season"] == season)]
            res, _ = run_regression(sub, "delta_SIA", "wind_stress")
            if res:
                row = {"sector": sector, "period": period, "season": season}
                row.update(res)
                regression_results.append(row)

regression_df = pd.DataFrame(regression_results)
regression_df.to_csv(f"{OUT_DIR}/wind_stress_regressions.csv", index=False)
print(f"Saved {len(regression_df)} regression results")
print(regression_df.head(10))

# ---------------- Check: has beta (sensitivity) changed pre vs post 2016? ----------------
print("\n" + "=" * 60)
print("CHECK: beta (sensitivity) and residual variance, pre vs post 2016")
print("=" * 60)
pivot_beta = regression_df.pivot_table(
    index=["sector", "season"], columns="period", values="beta"
)
pivot_beta["beta_change_pct"] = 100 * (pivot_beta["post_2016"] - pivot_beta["pre_2016"]) / pivot_beta["pre_2016"].abs()

pivot_resvar = regression_df.pivot_table(
    index=["sector", "season"], columns="period", values="residual_variance"
)
pivot_resvar["resvar_change_pct"] = 100 * (pivot_resvar["post_2016"] - pivot_resvar["pre_2016"]) / pivot_resvar["pre_2016"]

print("\nSensitivity (beta) change:")
print(pivot_beta)
print("\nResidual variance change (buffering diagnostic):")
print(pivot_resvar)

pivot_beta.to_csv(f"{OUT_DIR}/beta_pre_post_comparison.csv")
pivot_resvar.to_csv(f"{OUT_DIR}/residual_variance_pre_post_comparison.csv")

# ---------------- 2. AR(1) persistence model ----------------
print("\n" + "=" * 60)
print("AR(1) PERSISTENCE MODEL: X_t+1 = rho * X_t + noise")
print("=" * 60)

ar1_results = []
for sector in sectors:
    for period in ["pre_2016", "post_2016"]:
        sub = df[(df["sector"] == sector) & (df["period"] == period)].sort_values("date")
        sub = sub.dropna(subset=["SIA"])
        if len(sub) < 30:
            continue
        x_t = sub["SIA"].values[:-1]
        x_t1 = sub["SIA"].values[1:]
        X = sm.add_constant(x_t)
        model = sm.OLS(x_t1, X).fit()
        rho = model.params[1]
        ar1_results.append({
            "sector": sector, "period": period,
            "rho": rho, "one_minus_rho": 1 - rho,
            "r_squared": model.rsquared, "n_obs": len(sub),
        })
        print(f"  {sector} ({period}): rho = {rho:.4f}, R^2 = {model.rsquared:.4f}")

ar1_df = pd.DataFrame(ar1_results)
ar1_df.to_csv(f"{OUT_DIR}/ar1_persistence.csv", index=False)

pivot_rho = ar1_df.pivot_table(index="sector", columns="period", values="rho")
pivot_rho["rho_change"] = pivot_rho["post_2016"] - pivot_rho["pre_2016"]
print("\nPersistence (rho) change, pre vs post 2016:")
print(pivot_rho)
pivot_rho.to_csv(f"{OUT_DIR}/rho_pre_post_comparison.csv")

print(f"\n\nAll results saved to: {OUT_DIR}/")
print("Files: wind_stress_regressions.csv, beta_pre_post_comparison.csv,")
print("       residual_variance_pre_post_comparison.csv, ar1_persistence.csv,")
print("       rho_pre_post_comparison.csv")
