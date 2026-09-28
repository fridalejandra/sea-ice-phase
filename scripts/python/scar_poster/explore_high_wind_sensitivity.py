"""
explore_high_wind_sensitivity.py   [EXPLORATORY -- option 2, cheap version]

Does sensitivity behave differently for the STRONGEST wind events?

Tests 1-3 fit a mean regression over all days. A mean slope can be unchanged
while the tail behaves differently. This is the cheapest possible look at
that: refit the same interaction model on high-wind days only, and compare
to the all-days fit.

NOT a matched-event design. No event definition, no duration/geometry
matching, no storm tracking. Just a percentile subset. Treat as a first
look that says whether the full design is worth building.

Seasons are POOLED (per sector, not per sector x season) so the high-wind
subset keeps enough rows to estimate an interaction.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

DIV_PATH = "ice_divergence_by_sector_season.csv"
WIND_PATH = ("/user/geog/falejandraperez/sea-ice-phase/data/merged/"
             "analysis_table_daily_anomaly_periodclim.csv")
WIND_COL = "wind_stress_anomaly"
DIV_VAR = "div_positive"          # lead-opening: the cleanest null in Test 2

SPLIT_YEAR = 2016
EXCLUDE_YEARS = [1978, 1987, 1991, 1995]
PERCENTILES = [0.0, 0.75, 0.90, 0.95]   # 0.0 = all days (baseline)

SECTOR_RENAME = {"ABS": "Amundsen-Bellingshausen", "WED": "Weddell",
                 "KHV": "King Haakon VII", "EA": "East Antarctica",
                 "RA": "Ross-Amundsen"}
SEC_PER_DAY = 86400.0
OUT = "explore_high_wind_sensitivity.csv"


def main():
    div = pd.read_csv(DIV_PATH, parse_dates=["date"])
    wind = pd.read_csv(WIND_PATH, parse_dates=["date"])
    div["sector"] = div["sector"].replace(SECTOR_RENAME)
    if "n_valid_cells" in div:
        div = div[div.n_valid_cells > 0]
    wind = wind[["date", "sector", WIND_COL]].rename(columns={WIND_COL: "w"})

    df = div.merge(wind, on=["date", "sector"], how="inner")
    df = df[~df.year.isin(EXCLUDE_YEARS)]
    df["post"] = (df.year >= SPLIT_YEAR).astype(int)
    df["doy"] = df.date.dt.dayofyear

    # period-specific deseasonalisation of the response (as elsewhere)
    df["y"] = np.nan
    for s in df.sector.unique():
        for p in (0, 1):
            m = (df.sector == s) & (df.post == p)
            clim = df.loc[m].groupby("doy")[DIV_VAR].transform("mean")
            df.loc[m, "y"] = df.loc[m, DIV_VAR] - clim

    rows = []
    for sector in sorted(df.sector.unique()):
        sub_all = df[df.sector == sector].dropna(subset=["y", "w"])
        for pct in PERCENTILES:
            if pct == 0.0:
                sub = sub_all
                label = "all days"
            else:
                thr = sub_all.w.abs().quantile(pct)
                sub = sub_all[sub_all.w.abs() >= thr]
                label = f"|wind| >= p{int(pct*100)}"

            if sub.post.nunique() < 2 or len(sub) < 200:
                print(f"[skip] {sector} {label}: n={len(sub)}")
                continue

            m = smf.ols("y ~ w * post", data=sub).fit()
            rows.append({
                "sector": sector, "subset": label, "n": len(sub),
                "n_pre": int((sub.post == 0).sum()),
                "n_post": int((sub.post == 1).sum()),
                "beta_pre": m.params.get("w", np.nan) * SEC_PER_DAY,
                "interaction": m.params.get("w:post", np.nan) * SEC_PER_DAY,
                "p_interaction": m.pvalues.get("w:post", np.nan),
                "r2": m.rsquared,
            })

    res = pd.DataFrame(rows)
    res.to_csv(OUT, index=False)
    pd.set_option("display.width", 160)
    print(res.round(4).to_string(index=False))
    print(f"\n-> {OUT}")
    print("\nRead: does `interaction` grow (or become significant) as the "
          "subset tightens to stronger winds? If yes, the tail behaves "
          "differently from the mean and the matched-event design is worth "
          "building. If it stays flat and null across percentiles, the mean "
          "result holds in the tail too.")
    print("NOTE: p-values here are UNCORRECTED and the subsets are nested "
          "(not independent). Exploratory only -- do not report as a test.")


if __name__ == "__main__":
    main()
