"""
wind_sensitivity_interaction_test.py

Wind stress sensitivity (beta) pre/post-2016 shift test
=========================================================

Tests whether the sensitivity of daily sea ice area change to wind stress
has shifted across the 2016 regime shift, per sector and season/month.

Uses a single interaction-term regression per sector/period-bin rather
than fitting two separate OLS regressions and comparing slopes
informally. The interaction coefficient IS the estimated shift in beta,
and its block-bootstrap confidence interval / p-value directly answers
"did sensitivity change" for that sector/bin.

Two resolutions are run:
  - SEASONAL (DJF/MAM/JJA/SON): primary, confirmatory. Full FDR correction
    across all 5 sectors x 4 seasons = 20 tests.
  - MONTHLY (Jan..Dec): exploratory/diagnostic only. No FDR correction
    applied - p-values here are suggestive (used e.g. to check whether a
    seasonal signal is concentrated in one month, a la Kusahara's
    November effect), not standalone claims. 5 sectors x 12 months = 60
    tests, deliberately NOT corrected - label these as exploratory
    wherever they're reported.

Expects a dataframe (analysis_table_daily_anomaly.csv) with columns:
    date                 (datetime)
    sector               (str - one of the 5 real sector names, e.g.
                          "Weddell", "King Haakon VII", "East Antarctica",
                          "Ross-Amundsen", "Amundsen-Bellingshausen")
    wind_stress          (float, daily wind stress magnitude, RAW - not
                          the deseasonalized anomaly. Wind stress is the
                          forcing variable; its physical magnitude is
                          what mechanically drives the ice response, not
                          its deviation from typical-for-that-day.)
    delta_SIA_anomaly    (float, daily change in deseasonalized SIA
                          anomaly - the actual response variable, already
                          computed by deseasonalize_sia_and_wind.py)

Verified against synthetic data with known true beta_pre/beta_post before
being pointed at real data - see conversation history for the three test
cases (true shift detected, no shift correctly not detected, sign-flipped
shift correctly detected and signed).

NOTE on interpretation: this tests a SLOPE change (sensitivity), which is
conceptually distinct from a LEVEL/trend change in wind stress itself
(that's what trend_analysis_sector_month_season.py tests, via Mann-
Kendall). A sector can show one, both, or neither. If a sector shows a
significant beta_shift here AND also has a strong monotonic wind-stress
trend from the Mann-Kendall analysis, it's worth double-checking the
beta_shift is a genuine slope change and not an estimation-precision
artifact from unequal wind_stress spread between the pre/post subsets.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

# ---------------------------------------------------------------------
# 0. Config
# ---------------------------------------------------------------------

IN_CSV = (
    "/user/geog/falejandraperez/sea-ice-phase/data/merged/"
    "analysis_table_daily_anomaly.csv"
)
OUT_DIR = "/user/geog/falejandraperez/sea-ice-phase/data/merged/analysis_results/"

DATE_COL = "date"
SECTOR_COL = "sector"
WIND_COL = "wind_stress"          # RAW magnitude, not the anomaly - see note above
RESPONSE_COL = "delta_SIA_anomaly"

REGIME_SHIFT_YEAR = 2016
N_BOOTSTRAP = 500
BLOCK_YEARS = 3          # matches the AR(1) block-bootstrap choice used elsewhere
FDR_ALPHA = 0.05

SEASON_MAP = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJA", 7: "JJA", 8: "JJA",
    9: "SON", 10: "SON", 11: "SON",
}


# ---------------------------------------------------------------------
# 1. Core interaction-term regression for one sector/bin subset
# ---------------------------------------------------------------------

def fit_interaction_model(df_subset):
    """
    Fit: delta_SIA_anomaly ~ wind_stress + post + wind_stress:post

    The coefficient on wind_stress:post is the estimated shift in beta
    (post-2016 sensitivity minus pre-2016 sensitivity).

    Returns a dict of point estimates. Standard errors / p-values from
    this OLS fit are NOT used for inference - they assume iid residuals,
    which daily sea-ice anomalies violate (autocorrelation). Use
    block_bootstrap_interaction() below for actual inference.
    """
    df_subset = df_subset.dropna(subset=[WIND_COL, RESPONSE_COL])
    if len(df_subset) < 20:
        return None

    post = (df_subset[DATE_COL].dt.year >= REGIME_SHIFT_YEAR).astype(float)
    X = pd.DataFrame({
        WIND_COL: df_subset[WIND_COL].values,
        "post": post.values,
        f"{WIND_COL}_x_post": df_subset[WIND_COL].values * post.values,
    })
    X = sm.add_constant(X)
    y = df_subset[RESPONSE_COL].values

    model = sm.OLS(y, X).fit()

    return {
        "beta_pre": model.params[WIND_COL],
        "beta_shift": model.params[f"{WIND_COL}_x_post"],
        "beta_post": model.params[WIND_COL] + model.params[f"{WIND_COL}_x_post"],
        "r2": model.rsquared,
        "n_obs": len(df_subset),
    }


# ---------------------------------------------------------------------
# 2. Block bootstrap for the interaction term specifically
# ---------------------------------------------------------------------

def block_bootstrap_interaction(df_subset, n_boot=N_BOOTSTRAP, block_years=BLOCK_YEARS,
                                  seed=None):
    """
    Block-bootstrap CI and empirical p-value for beta_shift (the
    interaction coefficient). Same block-resampling logic as the AR(1)
    persistence CIs elsewhere in the pipeline - same rationale: daily
    residuals are autocorrelated, so naive OLS SEs on the interaction
    term would be too small and overstate significance.

    Blocks are drawn by contiguous ~block_years-year chunks of the
    subset's own date range, resampled with replacement, concatenated
    to the original subset length, refit each time.
    """
    rng = np.random.default_rng(seed)
    df_subset = df_subset.dropna(subset=[WIND_COL, RESPONSE_COL]).reset_index(drop=True)
    df_subset = df_subset.sort_values(DATE_COL).reset_index(drop=True)

    years = df_subset[DATE_COL].dt.year
    year_min, year_max = years.min(), years.max()
    block_starts = list(range(year_min, year_max + 1, block_years))

    blocks = []
    for start in block_starts:
        mask = (years >= start) & (years < start + block_years)
        if mask.sum() > 0:
            blocks.append(df_subset[mask])

    if len(blocks) < 4:
        # too few blocks to bootstrap meaningfully
        return None

    shifts = []
    for _ in range(n_boot):
        sampled_blocks = [blocks[i] for i in rng.integers(0, len(blocks), size=len(blocks))]
        resampled = pd.concat(sampled_blocks, ignore_index=True)
        fit = fit_interaction_model(resampled)
        if fit is not None:
            shifts.append(fit["beta_shift"])

    if len(shifts) < n_boot * 0.5:
        return None

    shifts = np.array(shifts)
    ci_low, ci_high = np.percentile(shifts, [2.5, 97.5])
    # two-sided empirical p-value: proportion of bootstrap shifts crossing zero
    p_value = 2 * min((shifts > 0).mean(), (shifts < 0).mean())
    p_value = min(p_value, 1.0)

    return {
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
        "significant_uncorrected": not (ci_low < 0 < ci_high),
    }


# ---------------------------------------------------------------------
# 3. Run across all sector x bin combinations
# ---------------------------------------------------------------------

def run_all_tests(df, bin_col="season", apply_fdr=True, fdr_alpha=FDR_ALPHA, seed=42):
    """
    Loops over every sector x bin (season or month) combination, fits
    the interaction model, bootstraps the interaction term, and
    (optionally) applies Benjamini-Hochberg FDR correction across all
    tests run in this call.

    Set bin_col='season' for the primary confirmatory pass (apply_fdr=True).
    Set bin_col='month' for the exploratory pass (apply_fdr=False) - keep
    monthly p-values labeled as uncorrected/exploratory wherever reported.
    """
    results = []
    sectors = df[SECTOR_COL].unique()
    bins_ = df[bin_col].unique()

    for sector in sectors:
        for b in bins_:
            subset = df[(df[SECTOR_COL] == sector) & (df[bin_col] == b)]
            fit = fit_interaction_model(subset)
            if fit is None:
                continue
            boot = block_bootstrap_interaction(subset, seed=seed)
            if boot is None:
                continue

            results.append({
                "sector": sector,
                bin_col: b,
                **fit,
                **boot,
            })

    results_df = pd.DataFrame(results)

    if apply_fdr and len(results_df) > 0:
        rejected, p_adj, _, _ = multipletests(
            results_df["p_value"], alpha=fdr_alpha, method="fdr_bh"
        )
        results_df["p_value_fdr"] = p_adj
        results_df["significant_fdr"] = rejected
    else:
        results_df["p_value_fdr"] = np.nan
        results_df["significant_fdr"] = False
        if not apply_fdr:
            results_df.rename(columns={"p_value": "p_value_exploratory"}, inplace=True)

    return results_df


# ---------------------------------------------------------------------
# 4. Run
# ---------------------------------------------------------------------

if __name__ == "__main__":
    df = pd.read_csv(IN_CSV, parse_dates=[DATE_COL])

    df["season"] = df[DATE_COL].dt.month.map(SEASON_MAP)
    df["month"] = df[DATE_COL].dt.strftime("%b")

    # --- Primary: seasonal, FDR-corrected ---
    print("Running seasonal (FDR-corrected) beta-shift tests...")
    seasonal_results = run_all_tests(df, bin_col="season", apply_fdr=True)
    seasonal_results.to_csv(OUT_DIR + "beta_shift_seasonal_fdr.csv", index=False)
    print(seasonal_results[["sector", "season", "beta_pre", "beta_post",
                             "beta_shift", "p_value_fdr", "significant_fdr"]])

    # --- Exploratory: monthly, uncorrected ---
    print("\nRunning monthly (exploratory, uncorrected) beta-shift tests...")
    monthly_results = run_all_tests(df, bin_col="month", apply_fdr=False)
    monthly_results.to_csv(OUT_DIR + "beta_shift_monthly_exploratory.csv", index=False)
    print(monthly_results[["sector", "month", "beta_pre", "beta_post",
                            "beta_shift", "p_value_exploratory"]])

    print(f"\nSaved both result tables to: {OUT_DIR}")