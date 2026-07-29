"""
persistence_efold_test.py

Tests whether the MEMORY/DECORRELATION TIMESCALE of r_t - the residual
of the interaction-term regression, i.e. daily tendency NOT explained by
wind stress - has shifted across the 2016 regime shift, by sector.

CORRECTED VERSION: earlier versions of this test ran on SIA_anomaly (the
LEVEL, X_t) directly. That was the wrong quantity. Your Ch4 equation is:
    X_t+1 = X_t + (dX/dt)|atmos + r
and your own chapter text is explicit that persistence is about how r -
"oceanic forcing or internal sea ice dynamics" - reinforces or dampens
over time, NOT about the autocorrelation of X_t itself. Testing X_t
directly is actively misleading: X_t is a RUNNING SUM of daily
tendencies, and any accumulated/integrated series shows artificially
long, slow-decaying autocorrelation almost mechanically, regardless of
the true memory of the underlying daily increments - the same reason you
would never test a random walk's memory by looking at the walk itself
rather than its increments. This is a likely major contributor to the
implausibly long/NaN e-folding times found when SIA_anomaly was tested
directly.

r_t is produced by extract_interaction_residuals.py: the residual of
the SAME interaction-term regression used in
wind_sensitivity_interaction_test.py (delta_SIA_anomaly ~ wind_stress +
post + wind_stress:post), fit per sector x season, concatenated into one
continuous daily series per sector. Because r_t is already a residual of
a model fit on delta_SIA_anomaly (a DIFFERENCED quantity, not a level),
it does not have the integrated-series artifact problem - this is the
textbook-correct way to test whether "shocks" to the system have their
own memory, independent of that artifact.

Method (unchanged from before, just pointed at r_t instead of X_t):
  1. Compute the autocorrelation function (ACF) of r_t out to MAX_LAG
     days, separately for pre-2016 and post-2016, per sector.
  2. Extract the e-folding timescale: the first lag at which ACF drops
     below 1/e (~0.368).
  3. Block-bootstrap (same 3yr blocks used elsewhere in this pipeline)
     to get a CI and p-value on the SHIFT in e-folding time pre vs post.

CAVEAT (read before trusting the CI): block-bootstrapping is well-
justified for a regression coefficient like beta (each day contributes
one row to a sum, and blocks just resample which years' worth of rows go
into that sum). It's less clean for the ACF itself - concatenating
resampled blocks introduces a small number of artificial "seams" where
two non-adjacent original days become adjacent in the resampled series,
which could locally distort short-lag correlation. With ~12 blocks over
a ~35+ year period, this affects a small fraction of the ~10,000+ lag
pairs and is likely a minor bias, but it is a real approximation, not an
exact method - flag this if presenting the CI as more precise than it is.

Verified against synthetic AR(1) data with known true e-folding time
before running on real data (see conversation history).
"""

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

IN_CSV = (
    "/user/geog/falejandraperez/sea-ice-phase/data/merged/analysis_results/"
    "interaction_residuals_daily.csv"
)
OUT_DIR = "/user/geog/falejandraperez/sea-ice-phase/data/merged/analysis_results/"

DATE_COL = "date"
SECTOR_COL = "sector"
RESPONSE_COL = "residual"   # r_t from extract_interaction_residuals.py -
                             # NOT SIA_anomaly. See docstring for why.

REGIME_SHIFT_YEAR = 2016
MAX_LAG = 60           # days - raised from an initial 30. Your original ACF
                        # diagnostic found autocorrelation still at 0.80-0.91
                        # AT LAG 10 (nowhere near the 1/e~0.368 threshold),
                        # so real e-folding time could plausibly exceed 30
                        # days. 60 gives real margin; if efold_pre/post still
                        # come back NaN for a sector, raise this further -
                        # that result itself would be informative (memory
                        # longer than 2 months in that sector).
N_BOOTSTRAP = 500
BLOCK_YEARS = 3
FDR_ALPHA = 0.05


# ---------------------------------------------------------------------
# 1. ACF and e-folding timescale
# ---------------------------------------------------------------------

def acf(x, max_lag):
    """
    Sample autocorrelation function, lags 0..max_lag.
    x: 1D array, assumed already gap-free/contiguous for this purpose
       (small gaps from missing days are tolerated via nan-aware mean/var,
       but large gaps will bias lagged pairs - fine here since MAM/JJA/etc
       subsets aren't used, this runs on the full contiguous daily series
       per period).
    """
    x = np.asarray(x, dtype=float)
    x = x - np.nanmean(x)
    n = len(x)
    var = np.nanvar(x)
    if var == 0 or np.isnan(var):
        return np.full(max_lag + 1, np.nan)

    result = np.empty(max_lag + 1)
    for lag in range(max_lag + 1):
        if lag == 0:
            result[0] = 1.0
            continue
        x1 = x[:-lag]
        x2 = x[lag:]
        mask = ~(np.isnan(x1) | np.isnan(x2))
        if mask.sum() < 10:
            result[lag] = np.nan
            continue
        result[lag] = np.mean(x1[mask] * x2[mask]) / var

    return result


def efold_timescale(acf_values):
    """
    First lag at which ACF drops below 1/e (~0.368). Returns np.nan if
    the ACF never drops below that threshold within the computed lags
    (i.e. memory extends beyond MAX_LAG - report this honestly rather
    than extrapolating).
    """
    threshold = 1 / np.e
    below = np.where(acf_values < threshold)[0]
    if len(below) == 0:
        return np.nan
    return float(below[0])


def compute_period_efold(df_subset, max_lag=MAX_LAG, calendar_align=True):
    """
    Compute ACF and e-folding time for a subset of rows.

    calendar_align=True (default, use for REAL data): reindexes onto a
    full daily calendar so any true gaps become NaN and lag arithmetic
    stays calendar-correct.

    calendar_align=False (use for BOOTSTRAP-RESAMPLED blocks): skips
    reindexing and just uses the values in their given row order. This
    is required for resampled data because block bootstrap deliberately
    reorders/duplicates blocks, so dates are no longer unique or
    monotonic - reindexing by calendar date on resampled data would
    either crash (duplicate dates) or silently misalign values.
    """
    df_subset = df_subset.dropna(subset=[RESPONSE_COL]).sort_values(DATE_COL)
    if len(df_subset) < 100:
        return None

    if calendar_align:
        full_range = pd.date_range(df_subset[DATE_COL].min(), df_subset[DATE_COL].max(), freq="D")
        series_values = df_subset.set_index(DATE_COL)[RESPONSE_COL].reindex(full_range).values
    else:
        series_values = df_subset[RESPONSE_COL].values

    acf_vals = acf(series_values, max_lag)
    efold = efold_timescale(acf_vals)
    return {"acf": acf_vals, "efold": efold, "n_obs": len(df_subset)}


# ---------------------------------------------------------------------
# 2. Block bootstrap for the e-folding SHIFT
# ---------------------------------------------------------------------

def block_bootstrap_efold_shift(df_subset, n_boot=N_BOOTSTRAP, block_years=BLOCK_YEARS, seed=None):
    """
    Block-bootstrap CI/p-value for (post e-fold - pre e-fold). Resamples
    ~block_years-year blocks WITHIN each period separately (so pre-2016
    blocks only ever get reassembled from pre-2016 years, and likewise
    for post), preserving the pre/post split in every bootstrap draw.
    """
    rng = np.random.default_rng(seed)
    df_subset = df_subset.dropna(subset=[RESPONSE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    df_subset["year"] = df_subset[DATE_COL].dt.year
    df_subset["period"] = np.where(df_subset["year"] >= REGIME_SHIFT_YEAR, "post", "pre")

    period_blocks = {}
    for period in ["pre", "post"]:
        sub = df_subset[df_subset["period"] == period]
        years = sub["year"]
        y_min, y_max = years.min(), years.max()
        blocks = []
        for start in range(y_min, y_max + 1, block_years):
            mask = (years >= start) & (years < start + block_years)
            if mask.sum() > 0:
                blocks.append(sub[mask])
        period_blocks[period] = blocks

    if len(period_blocks["pre"]) < 4 or len(period_blocks["post"]) < 2:
        return None

    shifts = []
    for _ in range(n_boot):
        resampled = {}
        for period in ["pre", "post"]:
            blocks = period_blocks[period]
            sampled = [blocks[i] for i in rng.integers(0, len(blocks), size=len(blocks))]
            resampled[period] = pd.concat(sampled, ignore_index=True)

        pre_result = compute_period_efold(resampled["pre"], calendar_align=False)
        post_result = compute_period_efold(resampled["post"], calendar_align=False)
        if pre_result is None or post_result is None:
            continue
        if np.isnan(pre_result["efold"]) or np.isnan(post_result["efold"]):
            continue
        shifts.append(post_result["efold"] - pre_result["efold"])

    if len(shifts) < n_boot * 0.5:
        return None

    shifts = np.array(shifts)
    ci_low, ci_high = np.percentile(shifts, [2.5, 97.5])
    p_value = 2 * min((shifts > 0).mean(), (shifts < 0).mean())
    p_value = min(p_value, 1.0)

    return {
        "ci_low": ci_low, "ci_high": ci_high, "p_value": p_value,
        "significant_uncorrected": not (ci_low < 0 < ci_high),
        "n_bootstrap_used": len(shifts),
    }


# ---------------------------------------------------------------------
# 3. Run per sector
# ---------------------------------------------------------------------

def run_efold_tests(df):
    results = []
    acf_curves = []

    for sector in df[SECTOR_COL].unique():
        sub = df[df[SECTOR_COL] == sector].copy()
        sub["year"] = sub[DATE_COL].dt.year
        sub["period"] = np.where(sub["year"] >= REGIME_SHIFT_YEAR, "post", "pre")

        pre = compute_period_efold(sub[sub["period"] == "pre"])
        post = compute_period_efold(sub[sub["period"] == "post"])

        if pre is None or post is None:
            print(f"  [{sector}] SKIPPED: not enough data in pre or post period "
                  f"(pre={'OK' if pre else 'insufficient (<100 obs)'}, "
                  f"post={'OK' if post else 'insufficient (<100 obs)'})")
            continue

        print(f"  [{sector}] real efold_pre={pre['efold']}, real efold_post={post['efold']} "
              f"(NaN means ACF never dropped below 1/e within MAX_LAG={MAX_LAG} days)")

        row = {
            "sector": sector,
            "efold_pre": pre["efold"],
            "efold_post": post["efold"],
            "efold_shift": (post["efold"] - pre["efold"]
                             if not (np.isnan(pre["efold"]) or np.isnan(post["efold"]))
                             else np.nan),
            "n_obs_pre": pre["n_obs"],
            "n_obs_post": post["n_obs"],
            "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan,
            "significant_uncorrected": False, "n_bootstrap_used": 0,
        }

        # Only attempt the bootstrap if the REAL data produced finite
        # e-folding values in both periods - if the real series can't
        # even produce a number, the bootstrap won't either.
        if not (np.isnan(pre["efold"]) or np.isnan(post["efold"])):
            boot = block_bootstrap_efold_shift(sub, seed=42)
            if boot is None:
                print(f"  [{sector}] bootstrap FAILED (too many resampled draws had "
                      f"NaN e-fold, or too few blocks) - real efold values above are "
                      f"still valid point estimates, just no CI/p-value.")
            else:
                row.update(boot)
        else:
            print(f"  [{sector}] bootstrap SKIPPED - real e-fold itself is NaN in at "
                  f"least one period, so no finite shift exists to bootstrap. This "
                  f"sector's memory genuinely exceeds MAX_LAG={MAX_LAG} days in that "
                  f"period - consider raising MAX_LAG if you want a number here.")

        results.append(row)

        for lag, val in enumerate(pre["acf"]):
            acf_curves.append({"sector": sector, "period": "pre", "lag": lag, "acf": val})
        for lag, val in enumerate(post["acf"]):
            acf_curves.append({"sector": sector, "period": "post", "lag": lag, "acf": val})

    results_df = pd.DataFrame(results)
    acf_df = pd.DataFrame(acf_curves)

    if len(results_df) > 0:
        valid = results_df["p_value"].notna()
        results_df["p_value_fdr"] = np.nan
        results_df["significant_fdr"] = False
        if valid.sum() > 0:
            rejected, p_adj, _, _ = multipletests(
                results_df.loc[valid, "p_value"], alpha=FDR_ALPHA, method="fdr_bh"
            )
            results_df.loc[valid, "p_value_fdr"] = p_adj
            results_df.loc[valid, "significant_fdr"] = rejected

    return results_df, acf_df


# ---------------------------------------------------------------------
# 4. Run
# ---------------------------------------------------------------------

if __name__ == "__main__":
    df = pd.read_csv(IN_CSV, parse_dates=[DATE_COL])

    print("Running e-folding decorrelation-timescale shift tests, per sector...")
    results_df, acf_df = run_efold_tests(df)

    results_df.to_csv(OUT_DIR + "persistence_efold_shift.csv", index=False)
    acf_df.to_csv(OUT_DIR + "persistence_acf_curves.csv", index=False)

    print(results_df[["sector", "efold_pre", "efold_post", "efold_shift",
                       "p_value_fdr", "significant_fdr"]])
    print(f"\nSaved shift results to: {OUT_DIR}persistence_efold_shift.csv")
    print(f"Saved full ACF curves (for plotting) to: {OUT_DIR}persistence_acf_curves.csv")
    print("\nNOTE: efold_pre/efold_post = NaN means the ACF never dropped "
          "below 1/e within MAX_LAG days for that period - i.e. memory "
          "extends beyond the tested window. Consider raising MAX_LAG if "
          "you see this.")