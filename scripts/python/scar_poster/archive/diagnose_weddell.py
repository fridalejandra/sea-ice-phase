"""
diagnose_weddell_mam.py

Follow-up diagnostic for the one FDR-significant result in the seasonal
beta-shift test: Weddell, MAM (beta_pre=-1,492,261 -> beta_post=-21,684,
i.e. sensitivity collapsed toward zero post-2016 - the OPPOSITE direction
from the "thinner ice = more sensitive" hypothesis).

Checks four things before trusting this as a real physical result:

1. Sample size asymmetry: pre-2016 is ~37 years of MAM, post-2016 is
   only ~9 years. A much smaller, noisier post-2016 sample could produce
   an unstable/near-zero slope estimate that isn't really "no
   relationship", just "not enough data to see it clearly."

2. R^2 in each period: does wind stress explain much variance in EITHER
   period? If R^2 is tiny in both, the "significant shift" may be a
   shift between two weak/noisy fits, not a meaningful mechanistic
   change.

3. Residual variance pre vs post: is the data itself noisier post-2016,
   independent of the wind-stress relationship?

4. Outlier/leverage check, with explicit attention to 2016-2017 (the
   Maud Rise polynya reopening years) even though that event's
   documented peak activity is JJA/SON, not MAM - checking empirically
   rather than assuming it does or doesn't matter for MAM specifically.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

IN_CSV = (
    "/user/geog/falejandraperez/sea-ice-phase/data/merged/"
    "analysis_table_daily_anomaly.csv"
)
DATE_COL = "date"
SECTOR_COL = "sector"
WIND_COL = "wind_stress"
RESPONSE_COL = "delta_SIA_anomaly"

SECTOR = "Weddell"
SEASON = "MAM"
REGIME_SHIFT_YEAR = 2016

SEASON_MAP = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJA", 7: "JJA", 8: "JJA",
    9: "SON", 10: "SON", 11: "SON",
}


def main():
    df = pd.read_csv(IN_CSV, parse_dates=[DATE_COL])
    df["season"] = df[DATE_COL].dt.month.map(SEASON_MAP)

    subset = df[(df[SECTOR_COL] == SECTOR) & (df["season"] == SEASON)].copy()
    subset = subset.dropna(subset=[WIND_COL, RESPONSE_COL])
    subset["year"] = subset[DATE_COL].dt.year
    subset["period"] = np.where(subset["year"] >= REGIME_SHIFT_YEAR, "post", "pre")

    print(f"=== {SECTOR}, {SEASON}: diagnostic breakdown ===\n")

    # ---- 1. Sample size asymmetry ----
    print("--- 1. Sample size / period length ---")
    for period in ["pre", "post"]:
        sub = subset[subset["period"] == period]
        n_years = sub["year"].nunique()
        n_days = len(sub)
        print(f"  {period}-2016: {n_years} years, {n_days} days "
              f"({sub['year'].min()}-{sub['year'].max()})")
    print()

    # ---- 2. R^2 in each period, fit separately ----
    print("--- 2. R^2 by period (does wind stress explain much variance either way?) ---")
    for period in ["pre", "post"]:
        sub = subset[subset["period"] == period]
        X = sm.add_constant(sub[WIND_COL].values)
        y = sub[RESPONSE_COL].values
        model = sm.OLS(y, X).fit()
        print(f"  {period}-2016: R^2={model.rsquared:.4f}, "
              f"slope={model.params[1]:,.0f}, n={len(sub)}")
    print()

    # ---- 3. Residual variance pre vs post ----
    print("--- 3. Residual variance of delta_SIA_anomaly (raw, not vs wind) ---")
    for period in ["pre", "post"]:
        sub = subset[subset["period"] == period]
        print(f"  {period}-2016: std={sub[RESPONSE_COL].std():,.0f}, "
              f"var={sub[RESPONSE_COL].var():,.0f}")
    print()

    # ---- 4. Outlier / leverage check, flagging 2016-2017 specifically ----
    print("--- 4. Top 10 most extreme |delta_SIA_anomaly| days in post-2016 MAM ---")
    post = subset[subset["period"] == "post"].copy()
    post["abs_response"] = post[RESPONSE_COL].abs()
    top10 = post.nlargest(10, "abs_response")[
        [DATE_COL, "year", WIND_COL, RESPONSE_COL]
    ]
    print(top10.to_string(index=False))
    print()

    n_2016_17 = post[post["year"].isin([2016, 2017])].shape[0]
    n_post_total = len(post)
    n_2016_17_in_top10 = top10[top10["year"].isin([2016, 2017])].shape[0]
    print(f"2016-2017 days make up {n_2016_17}/{n_post_total} "
          f"({100*n_2016_17/n_post_total:.1f}%) of the post-2016 MAM sample, "
          f"but {n_2016_17_in_top10}/10 of the top-10 most extreme days.")
    print("(If the second number is much higher than the first, 2016-2017 "
          "is disproportionately driving the extremes even in MAM, despite "
          "the polynya's documented peak activity being JJA/SON.)")


if __name__ == "__main__":
    main()