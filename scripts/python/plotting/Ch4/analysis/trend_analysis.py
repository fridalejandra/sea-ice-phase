"""
trend_analysis_sector_month_season.py

Mann-Kendall trend test + Sen's slope estimator for SIA and wind stress,
computed separately by sector x calendar month (60 tests: 5 sectors x 12
months) and sector x season (20 tests: 5 sectors x 4 seasons).

This is a DIAGNOSTIC pass over the full record - it asks "is there a
monotonic trend over time" independent of the 2016 regime-shift question.
Complementary to, not a replacement for, the interaction-term regression
in wind_sensitivity_interaction_test.py, which specifically tests whether
SENSITIVITY (beta) shifted across 2016.

Method: Mann-Kendall is non-parametric (no normality assumption) and
Sen's slope is the median of all pairwise slopes (robust to outliers) -
same method used in Radlwimmer et al. (2026) for their Ross Sea storm
index trends (Hamed-Rao modified Mann-Kendall at p<0.05).

Variables tested:
  - SIA_anomaly   (deseasonalized SIA anomaly, km^2)
  - wind_stress   (RAW magnitude, not the anomaly - testing whether the
                   forcing itself trended, matching the poster's "did
                   wind stress intensify" framing, not a deseasonalized
                   view)

NOTE: wind SPEED is not available anywhere in this pipeline -
build_forcing_sector_table.py only ever derives wind_stress (tau
magnitude) from ERA5 ewss/nsss. If you want speed specifically, that
needs a different ERA5 variable pulled fresh, not something derivable
from stress alone without assuming a drag coefficient.
"""

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

IN_CSV = (
    "/user/geog/falejandraperez/sea-ice-phase/data/merged/"
    "analysis_table_daily_anomaly.csv"
)
OUT_DIR = "/user/geog/falejandraperez/sea-ice-phase/data/merged/analysis_results/"

DATE_COL = "date"
SECTOR_COL = "sector"
FDR_ALPHA = 0.05

SEASON_MAP = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJA", 7: "JJA", 8: "JJA",
    9: "SON", 10: "SON", 11: "SON",
}


# ---------------------------------------------------------------------
# Mann-Kendall trend test + Sen's slope, implemented directly (no extra
# dependency beyond numpy/scipy/statsmodels, which you already have)
# ---------------------------------------------------------------------

def mann_kendall_test(x, y):
    """
    Standard Mann-Kendall trend test with normal approximation for the
    p-value (valid for n >= ~10, fine here since we're testing annual
    series of ~45 points).

    x: array of time points (e.g. years) - used only for Sen's slope,
       MK statistic itself only depends on the ORDER of y.
    y: array of values, same length as x, NaNs allowed (dropped first).

    Returns dict with: n, S, z, p_value, trend ('increasing',
    'decreasing', 'no trend'), sen_slope.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~np.isnan(y)
    x, y = x[mask], y[mask]
    n = len(y)

    if n < 8:
        return {
            "n": n, "S": np.nan, "z": np.nan, "p_value": np.nan,
            "trend": "insufficient_data", "sen_slope": np.nan,
        }

    # --- Mann-Kendall S statistic ---
    S = 0
    for i in range(n - 1):
        S += np.sum(np.sign(y[i + 1:] - y[i]))

    # --- variance, accounting for tied values ---
    unique_vals, counts = np.unique(y, return_counts=True)
    tie_term = np.sum(counts * (counts - 1) * (2 * counts + 5))
    var_S = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0

    if S > 0:
        z = (S - 1) / np.sqrt(var_S)
    elif S < 0:
        z = (S + 1) / np.sqrt(var_S)
    else:
        z = 0.0

    p_value = 2 * (1 - _norm_cdf(abs(z)))

    if p_value < FDR_ALPHA and z > 0:
        trend = "increasing"
    elif p_value < FDR_ALPHA and z < 0:
        trend = "decreasing"
    else:
        trend = "no trend"

    # --- Sen's slope: median of all pairwise slopes ---
    slopes = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            if x[j] != x[i]:
                slopes.append((y[j] - y[i]) / (x[j] - x[i]))
    sen_slope = np.median(slopes) if slopes else np.nan

    return {
        "n": n, "S": S, "z": z, "p_value": p_value,
        "trend": trend, "sen_slope": sen_slope,
    }


def _norm_cdf(z):
    """Standard normal CDF without requiring scipy.stats import at call site."""
    from scipy.stats import norm
    return norm.cdf(z)


# ---------------------------------------------------------------------
# Build annual series per sector x (month or season), then run MK test
# ---------------------------------------------------------------------

def run_trend_tests(df, value_col, bin_col):
    """
    For each sector x bin (month or season) combination, build an
    annual time series (mean of value_col within that bin, per year),
    then run Mann-Kendall + Sen's slope on that annual series.
    """
    results = []
    df = df.copy()
    df["year"] = df[DATE_COL].dt.year

    for sector in df[SECTOR_COL].unique():
        for b in df[bin_col].unique():
            subset = df[(df[SECTOR_COL] == sector) & (df[bin_col] == b)]
            annual = subset.groupby("year")[value_col].mean().reset_index()
            annual = annual.dropna()

            if len(annual) < 8:
                continue

            mk = mann_kendall_test(annual["year"].values, annual[value_col].values)
            results.append({
                "sector": sector,
                bin_col: b,
                "variable": value_col,
                **mk,
            })

    results_df = pd.DataFrame(results)

    if len(results_df) > 0:
        valid = results_df["p_value"].notna()
        rejected, p_adj, _, _ = multipletests(
            results_df.loc[valid, "p_value"], alpha=FDR_ALPHA, method="fdr_bh"
        )
        results_df.loc[valid, "p_value_fdr"] = p_adj
        results_df.loc[valid, "significant_fdr"] = rejected
        results_df["p_value_fdr"] = results_df["p_value_fdr"].fillna(np.nan)
        results_df["significant_fdr"] = results_df["significant_fdr"].fillna(False)

    return results_df


# ---------------------------------------------------------------------

if __name__ == "__main__":
    df = pd.read_csv(IN_CSV, parse_dates=[DATE_COL])
    df["month"] = df[DATE_COL].dt.strftime("%b")
    df["month_num"] = df[DATE_COL].dt.month
    df["season"] = df["month_num"].map(SEASON_MAP)

    print("Running SIA trend tests (sector x month)...")
    sia_month = run_trend_tests(df, "SIA_anomaly", "month")
    sia_month.to_csv(OUT_DIR + "trend_sia_sector_month.csv", index=False)
    print(sia_month[["sector", "month", "sen_slope", "p_value_fdr", "significant_fdr"]])

    print("\nRunning SIA trend tests (sector x season)...")
    sia_season = run_trend_tests(df, "SIA_anomaly", "season")
    sia_season.to_csv(OUT_DIR + "trend_sia_sector_season.csv", index=False)
    print(sia_season[["sector", "season", "sen_slope", "p_value_fdr", "significant_fdr"]])

    print("\nRunning wind stress trend tests (sector x month)...")
    wind_month = run_trend_tests(df, "wind_stress", "month")
    wind_month.to_csv(OUT_DIR + "trend_wind_sector_month.csv", index=False)
    print(wind_month[["sector", "month", "sen_slope", "p_value_fdr", "significant_fdr"]])

    print("\nRunning wind stress trend tests (sector x season)...")
    wind_season = run_trend_tests(df, "wind_stress", "season")
    wind_season.to_csv(OUT_DIR + "trend_wind_sector_season.csv", index=False)
    print(wind_season[["sector", "season", "sen_slope", "p_value_fdr", "significant_fdr"]])

    print(f"\nSaved all 4 result tables to: {OUT_DIR}")
    print("NOTE: wind speed not tested - not present anywhere in the pipeline, "
          "only wind_stress (magnitude of tau) is available.")