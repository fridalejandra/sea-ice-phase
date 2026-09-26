#!/usr/bin/env python
"""
06c_divergence_variance_trend_from1988.py -- Ch4: is day-to-day
divergence/convergence variability changing, and is it a STEP at 2016, a
TREND, or neither -- restricted to 1988+ to avoid the pre-1988 drift
artifact that contaminated the full/recent pre-post comparison?

Mirrors 06b_sia_variance_timeseries_from1988.py: reduce the field to one
sector-mean daily scalar, compute within-season-year variance about that
season-year's own mean, regress log2(variance) on year, and pick between
  M0 constant | M1 linear trend | M2 step at 2016 | M3 trend + step
by AIC. Reports the trend-adjusted step (M3) and the trend (M1) regardless
of which model wins, so numbers are comparable across sectors.
"""
import numpy as np
import pandas as pd
import xarray as xr
import statsmodels.formula.api as smf

ROOT = "/user/geog/falejandraperez/sea-ice-phase"
IN_NC = f"{ROOT}/results/ch4/derived_nc/ease_divergence_with_latlon.nc"
OUT_CSV = f"{ROOT}/results/ch4/tables/divergence_variance_step_vs_trend_from1988.csv"

START_YEAR = 1988
BREAK_YEAR = 2016
MIN_DAYS = 20
MIN_SEASON_YEARS = 10
MIN_POST_YEARS = 4

VARS = ["divergence", "div_positive", "div_negative"]
TIME_DIM = "time"
LAT_NAME = "lat"
LON_NAME = "lon"

SEASONS = {"JJA": (6, 7, 8), "SON": (9, 10, 11), "DJF": (12, 1, 2), "MAM": (3, 4, 5)}

SECTORS = {
    "WS":  (-60.0,   20.0),
    "KH":  ( 20.0,   90.0),
    "EA":  ( 90.0,  160.0),
    "RS":  (160.0, -130.0),
    "ABS": (-130.0, -60.0),
}


def sector_mask(lon, lo, hi):
    lon = ((lon + 180) % 360) - 180
    if lo <= hi:
        return (lon >= lo) & (lon < hi)
    return (lon >= lo) | (lon < hi)


def season_year(dt_index, season):
    if season == "DJF":
        return np.where(dt_index.month == 12, dt_index.year + 1, dt_index.year)
    return dt_index.year


def fit_models(vdf):
    df = vdf.copy()
    df["post"] = (df.year >= BREAK_YEAR).astype(int)
    df["year_c"] = df.year - df.year.mean()

    m0 = smf.ols("log2var ~ 1", data=df).fit()
    m1 = smf.ols("log2var ~ year_c", data=df).fit()
    m2 = smf.ols("log2var ~ post", data=df).fit()
    m3 = smf.ols("log2var ~ year_c + post", data=df).fit()

    aics = {"M0_const": m0.aic, "M1_trend": m1.aic, "M2_step": m2.aic,
            "M3_trend+step": m3.aic}
    best = min(aics, key=aics.get)
    s = sorted(aics.values())
    dAIC = s[1] - s[0]

    step_pct = (2 ** m3.params["post"] - 1) * 100
    step_p = m3.pvalues["post"]
    step_only_pct = (2 ** m2.params["post"] - 1) * 100
    step_only_p = m2.pvalues["post"]
    trend_pct_per_decade = (2 ** (m1.params["year_c"] * 10) - 1) * 100
    trend_p = m1.pvalues["year_c"]

    return dict(best_model=best, dAIC_vs_next=dAIC, **{f"AIC_{k}": v for k, v in aics.items()},
                step_only_pct=step_only_pct, step_only_p=step_only_p,
                step_trendadj_pct=step_pct, step_trendadj_p=step_p,
                trend_pct_per_decade=trend_pct_per_decade, trend_p=trend_p)


def main():
    ds = xr.open_dataset(IN_NC, chunks={"time": 500})
    lon = ds[LON_NAME]
    rows = []

    for var in VARS:
        da = ds[var]
        for sec, (lo, hi) in SECTORS.items():
            mask = sector_mask(lon, lo, hi)
            spatial_dims = [d for d in da.dims if d != TIME_DIM]
            daily = da.where(mask).mean(dim=spatial_dims, skipna=True)
            daily = daily.sel({TIME_DIM: daily[TIME_DIM].dt.year >= START_YEAR})

            df = daily.to_dataframe(name="value").reset_index().dropna(subset=["value"])
            dt_index = pd.DatetimeIndex(df[TIME_DIM])

            for season, months in SEASONS.items():
                sub = df[dt_index.month.isin(months)].copy()
                sub["syear"] = season_year(pd.DatetimeIndex(sub[TIME_DIM]), season)

                var_rows = []
                for yr, g in sub.groupby("syear"):
                    if len(g) < MIN_DAYS:
                        continue
                    v = g["value"].var(ddof=1)
                    if pd.notna(v) and v > 0:
                        var_rows.append((yr, v))

                vdf = pd.DataFrame(var_rows, columns=["year", "var"])
                n_post = int((vdf.year >= BREAK_YEAR).sum()) if len(vdf) else 0
                if len(vdf) < MIN_SEASON_YEARS or n_post < MIN_POST_YEARS:
                    continue

                ref = vdf.loc[vdf.year < BREAK_YEAR, "var"].mean()
                vdf["log2var"] = np.log2(vdf["var"] / ref)

                res = fit_models(vdf)
                res.update(variable=var, sector=sec, season=season,
                           n_years=len(vdf), n_post=n_post)
                rows.append(res)

    out = pd.DataFrame(rows)
    cols = ["variable", "sector", "season", "n_years", "n_post", "best_model",
            "dAIC_vs_next", "AIC_M0_const", "AIC_M1_trend", "AIC_M2_step",
            "AIC_M3_trend+step", "step_only_pct", "step_only_p",
            "step_trendadj_pct", "step_trendadj_p", "trend_pct_per_decade", "trend_p"]
    out = out[cols]
    out.to_csv(OUT_CSV, index=False, float_format="%.6g")
    print(f"wrote {OUT_CSV}: {len(out)} rows")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
