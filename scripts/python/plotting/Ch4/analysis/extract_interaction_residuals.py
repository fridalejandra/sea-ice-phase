"""
extract_interaction_residuals.py

Produces the actual quantity the Ch4 persistence framework asks about.

Your equation is:
    X_t+1 = X_t + (dX/dt)|atmos + r

Persistence, per your own chapter text, is about how r (the residual -
"oceanic forcing or internal sea ice dynamics", the part of the daily
tendency NOT explained by wind) reinforces or dampens over time - NOT
about the autocorrelation of X_t itself.

Testing ACF/e-folding directly on SIA_anomaly (X_t, the level) is a
different and misleading question: X_t is a RUNNING SUM of daily
tendencies (X_t = X_0 + sum of all prior (dX/dt)|atmos + sum of all
prior r). Any accumulated/integrated series shows artificially long,
slow-decaying autocorrelation almost mechanically, regardless of the
true memory of the underlying daily increments - the same reason you'd
never test a random walk's memory by looking at the walk itself rather
than its increments. This is very likely a real contributor to the
implausibly long/NaN e-folding times found when testing SIA_anomaly
directly, separate from (and possibly more important than) the
climatology-drift issue found earlier.

This script extracts r_t properly: for each sector x season, refits the
SAME interaction-term regression already used in
wind_sensitivity_interaction_test.py (delta_SIA_anomaly ~ wind_stress +
post + wind_stress:post) and saves the RESIDUALS (with their dates),
instead of just the coefficients. Residuals from all 4 seasons are
concatenated per sector into one continuous, calendar-ordered daily
series - each day's residual computed using ITS OWN season's fitted
model, so the beta_pre/beta_post/interaction structure already
established for that season is properly divided out before what's left
(r_t) gets tested for its own persistence.

Built-in sanity property: OLS residuals from a model WITH an intercept
always sum to exactly zero by construction - this is checked and printed
as a basic correctness guard, not a meaningful new finding.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

IN_CSV = (
    "/user/geog/falejandraperez/sea-ice-phase/data/merged/"
    "analysis_table_daily_anomaly_periodclim.csv"
)
OUT_CSV = (
    "/user/geog/falejandraperez/sea-ice-phase/data/merged/analysis_results/"
    "interaction_residuals_daily.csv"
)

DATE_COL = "date"
SECTOR_COL = "sector"
WIND_COL = "wind_stress"
RESPONSE_COL = "delta_SIA_anomaly"
REGIME_SHIFT_YEAR = 2016

SEASON_MAP = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJA", 7: "JJA", 8: "JJA",
    9: "SON", 10: "SON", 11: "SON",
}


def fit_and_get_residuals(df_subset):
    """
    Same model as wind_sensitivity_interaction_test.py:
        delta_SIA_anomaly ~ wind_stress + post + wind_stress:post
    but returns per-row residuals (with dates) instead of just
    coefficients.
    """
    df_subset = df_subset.dropna(subset=[WIND_COL, RESPONSE_COL]).copy()
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
    residuals = y - model.predict(X)

    return pd.DataFrame({
        DATE_COL: df_subset[DATE_COL].values,
        "residual": residuals,
    })


def extract_all_residuals(df):
    all_residuals = []

    for sector in df[SECTOR_COL].unique():
        sector_residuals = []
        for season in df["season"].unique():
            subset = df[(df[SECTOR_COL] == sector) & (df["season"] == season)]
            resid_df = fit_and_get_residuals(subset)
            if resid_df is None:
                print(f"  [{sector}, {season}] SKIPPED - not enough data")
                continue

            resid_mean = resid_df["residual"].mean()
            print(f"  [{sector}, {season}] n={len(resid_df)}, "
                  f"residual mean={resid_mean:.6e} (should be ~0, OLS property)")

            sector_residuals.append(resid_df)

        if sector_residuals:
            sector_df = pd.concat(sector_residuals, ignore_index=True)
            sector_df[SECTOR_COL] = sector
            sector_df = sector_df.sort_values(DATE_COL).reset_index(drop=True)
            all_residuals.append(sector_df)

    return pd.concat(all_residuals, ignore_index=True)


if __name__ == "__main__":
    df = pd.read_csv(IN_CSV, parse_dates=[DATE_COL])
    df["season"] = df[DATE_COL].dt.month.map(SEASON_MAP)

    print("Extracting interaction-model residuals (r_t) per sector x season...\n")
    residuals_df = extract_all_residuals(df)

    residuals_df = residuals_df[[DATE_COL, SECTOR_COL, "residual"]]
    residuals_df.to_csv(OUT_CSV, index=False)

    print(f"\nSaved {len(residuals_df)} residual rows to: {OUT_CSV}")
    print("\nRow count per sector (should roughly match original daily row counts, "
          "minus any dropped NaN days):")
    print(residuals_df.groupby(SECTOR_COL).size())