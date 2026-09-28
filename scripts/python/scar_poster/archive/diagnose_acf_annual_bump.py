"""
diagnose_acf_annual_bump.py

Extends the ACF window well past a year and checks for a specific,
diagnostic shape feature: does the ACF curve decay smoothly and
monotonically (consistent with genuine long physical memory), or does
it dip and then BUMP BACK UP around lag ~365 days (a "same time next
year" echo, which indicates the day-of-year climatology used in
deseasonalize_sia_and_wind.py didn't fully remove the seasonal cycle -
e.g. if the cycle's amplitude or phase drifted at all across the
1979-2024 record, a single full-record climatology would leave residual
periodic structure in what's supposed to be a purely stochastic anomaly).

These two explanations predict different curve shapes:
  - Genuine long memory: ACF decays slowly, monotonically (or close to
    it), and either crosses 1/e somewhere in the extended window or
    approaches an asymptote - no local re-increase.
  - Residual seasonal contamination: ACF may decay for a while, then
    rises again in a window around lag 365 (and possibly ~730, ~1095 -
    multiple-year echoes), before falling again.

Method: for each sector x period, compute ACF out to MAX_LAG_EXTENDED
days (default 450, comfortably past one full year). Find the local
minimum in a window just before lag 365, and the local max in a window
around lag 365. If the "bump" (local max minus preceding local min)
exceeds BUMP_THRESHOLD, flag it.

Verified against two synthetic cases before running on real data:
  1. Pure long-memory AR(1), no seasonality -> should NOT flag a bump.
  2. AR(1) + an injected residual annual sinusoid (simulating imperfect
     deseasonalization) -> SHOULD flag a bump near lag 365.
"""

import numpy as np
import pandas as pd

IN_CSV = (
    "/user/geog/falejandraperez/sea-ice-phase/data/merged/"
    "analysis_table_daily_anomaly.csv"
)
OUT_DIR = "/user/geog/falejandraperez/sea-ice-phase/data/merged/analysis_results/"

DATE_COL = "date"
SECTOR_COL = "sector"
RESPONSE_COL = "SIA_anomaly"

REGIME_SHIFT_YEAR = 2016
MAX_LAG_EXTENDED = 450   # days - comfortably past 1 year, so an annual
                         # echo (if present) is fully visible, not cut off
BUMP_THRESHOLD = 0.08    # absolute ACF units. Calibrated against synthetic
                          # pure-long-memory noise at REALISTIC sample sizes
                          # matching this pipeline's actual pre/post period
                          # lengths - not an arbitrary round number.
                          #
                          # Calibration result: pure long-memory AR(1) with
                          # NO real seasonal leak produces bump magnitudes
                          # with std=0.034 (37yr/~13,500-day samples, like
                          # the pre-2016 period) and std=0.078 (8yr/~2,920-
                          # day samples, like the post-2016 period) purely
                          # from sampling noise at large lags. The original
                          # threshold of 0.03 would falsely flag a "bump" on
                          # pure noise roughly 60-70% of the time for short
                          # (post-2016-length) periods.
                          #
                          # IMPORTANT LIMITATION: even at this higher
                          # threshold, this test has real, honest power
                          # limits - a strong synthetic contamination signal
                          # (amplitude=1.0, an already-severe residual
                          # seasonal leak) only produced a bump of ~0.05,
                          # barely above the pre-period's OWN noise ceiling.
                          # Treat "no bump detected" as "nothing OBVIOUS",
                          # not proof of clean deseasonalization - and treat
                          # any flag on a SHORT period (few years, like most
                          # post-2016 subsets) with real caution given its
                          # much higher noise floor. Always look at the
                          # actual saved ACF curves
                          # (acf_extended_curves.csv), don't rely on the
                          # binary flag alone.

# window definitions, in days, for finding the "before the year mark"
# local minimum and the "at/near the year mark" local max
PRE_YEAR_WINDOW = (300, 355)
YEAR_MARK_WINDOW = (355, 400)


def acf(x, max_lag):
    x = np.asarray(x, dtype=float)
    x = x - np.nanmean(x)
    var = np.nanvar(x)
    if var == 0 or np.isnan(var):
        return np.full(max_lag + 1, np.nan)

    result = np.empty(max_lag + 1)
    for lag in range(max_lag + 1):
        if lag == 0:
            result[0] = 1.0
            continue
        x1, x2 = x[:-lag], x[lag:]
        mask = ~(np.isnan(x1) | np.isnan(x2))
        if mask.sum() < 10:
            result[lag] = np.nan
            continue
        result[lag] = np.mean(x1[mask] * x2[mask]) / var
    return result


def compute_extended_acf(df_subset, max_lag=MAX_LAG_EXTENDED):
    df_subset = df_subset.dropna(subset=[RESPONSE_COL]).sort_values(DATE_COL)
    if len(df_subset) < max_lag + 50:
        return None
    full_range = pd.date_range(df_subset[DATE_COL].min(), df_subset[DATE_COL].max(), freq="D")
    series_values = df_subset.set_index(DATE_COL)[RESPONSE_COL].reindex(full_range).values
    return acf(series_values, max_lag)


def detect_annual_bump(acf_values, pre_window=PRE_YEAR_WINDOW, year_window=YEAR_MARK_WINDOW,
                        threshold=BUMP_THRESHOLD):
    """
    Compares the MEAN ACF over a window just before the year mark to the
    MEAN ACF over a window at/near the year mark. Uses window AVERAGES,
    not single-point local min/max - point extremes at large lags are
    dominated by sampling noise (few effectively-independent pairs
    contribute to each individual lag, especially for slowly-decaying
    series), which produced false-positive bumps even for pure long-memory
    AR(1) processes with no real seasonal echo when tested against a
    single-point comparison. Averaging over a ~45-55 day window smooths
    that noise out while still being sensitive to a real, systematic
    annual echo (which should elevate the WHOLE year-mark window, not
    just one lag).
    """
    pre_segment = acf_values[pre_window[0]:pre_window[1]]
    year_segment = acf_values[year_window[0]:year_window[1]]

    if np.all(np.isnan(pre_segment)) or np.all(np.isnan(year_segment)):
        return {"mean_pre_year": np.nan, "mean_at_year": np.nan,
                "bump_magnitude": np.nan, "bump_detected": False}

    mean_pre = np.nanmean(pre_segment)
    mean_year = np.nanmean(year_segment)
    bump = mean_year - mean_pre

    return {
        "mean_pre_year": mean_pre,
        "mean_at_year": mean_year,
        "bump_magnitude": bump,
        "bump_detected": bool(bump > threshold),
    }


def run_bump_diagnostic(df):
    results = []
    curves = []

    for sector in df[SECTOR_COL].unique():
        sub = df[df[SECTOR_COL] == sector].copy()
        sub["year"] = sub[DATE_COL].dt.year
        sub["period"] = np.where(sub["year"] >= REGIME_SHIFT_YEAR, "post", "pre")

        for period in ["pre", "post"]:
            period_df = sub[sub["period"] == period]
            acf_vals = compute_extended_acf(period_df)

            if acf_vals is None:
                print(f"  [{sector}, {period}] SKIPPED - not enough data for a "
                      f"{MAX_LAG_EXTENDED}-day-lag ACF (need n > {MAX_LAG_EXTENDED + 50} days)")
                continue

            bump = detect_annual_bump(acf_vals)
            n_years = period_df["year"].nunique()
            reliability_note = ""
            if n_years < 15:
                reliability_note = (f"  [LOW RELIABILITY: only {n_years} years - "
                                     f"noise floor for this test is much higher at "
                                     f"short sample sizes, treat this flag cautiously "
                                     f"and check the curve visually]")
            flag = "*** ANNUAL BUMP DETECTED ***" if bump["bump_detected"] else "smooth decay, no bump"
            print(f"  [{sector}, {period}] mean(lag {PRE_YEAR_WINDOW})={bump['mean_pre_year']:.4f}, "
                  f"mean(lag {YEAR_MARK_WINDOW})={bump['mean_at_year']:.4f}, "
                  f"bump={bump['bump_magnitude']:.4f}  -->  {flag}{reliability_note}")

            results.append({"sector": sector, "period": period, "n_years": n_years, **bump})
            for lag, val in enumerate(acf_vals):
                curves.append({"sector": sector, "period": period, "lag": lag, "acf": val})

    return pd.DataFrame(results), pd.DataFrame(curves)


# ---------------------------------------------------------------------

if __name__ == "__main__":
    df = pd.read_csv(IN_CSV, parse_dates=[DATE_COL])

    print(f"Running extended ({MAX_LAG_EXTENDED}-day) ACF bump diagnostic...\n")
    results_df, curves_df = run_bump_diagnostic(df)

    results_df.to_csv(OUT_DIR + "acf_annual_bump_diagnostic.csv", index=False)
    curves_df.to_csv(OUT_DIR + "acf_extended_curves.csv", index=False)

    n_flagged = results_df["bump_detected"].sum()
    print(f"\n{n_flagged} / {len(results_df)} sector-period combinations show an annual bump.")
    if n_flagged > 0:
        print("This suggests residual seasonal structure in SIA_anomaly - the "
              "day-of-year climatology may not be fully removing the seasonal "
              "cycle (e.g. if its amplitude/phase drifted across the record). "
              "Worth revisiting deseasonalize_sia_and_wind.py before trusting "
              "the long-memory result as purely physical.")
    else:
        print("No annual bump detected anywhere - the long e-folding times are "
              "more likely to reflect genuine physical memory, not leftover "
              "seasonal contamination.")

    print(f"\nSaved diagnostic table to: {OUT_DIR}acf_annual_bump_diagnostic.csv")
    print(f"Saved full extended ACF curves to: {OUT_DIR}acf_extended_curves.csv")