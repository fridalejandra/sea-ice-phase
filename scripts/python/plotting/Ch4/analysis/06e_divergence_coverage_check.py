#!/usr/bin/env python
"""
06e_divergence_coverage_check.py -- diagnostic: did the number/location of
valid (ice-covered, retrievable) grid cells per sector change from before to
after 2016? If so, a trend in sector-mean day-to-day variance could reflect
a shrinking/shifting sample of ice, not a change in how the remaining ice
moves.
"""
import numpy as np
import pandas as pd
import xarray as xr

ROOT = "/user/geog/falejandraperez/sea-ice-phase"
IN_NC = f"{ROOT}/results/ch4/derived_nc/ease_divergence_with_latlon.nc"
OUT_CSV = f"{ROOT}/results/ch4/tables/divergence_coverage_by_sector_season.csv"

START_YEAR = 1988
BREAK_YEAR = 2016
SEASONS = {"JJA": (6, 7, 8), "SON": (9, 10, 11), "DJF": (12, 1, 2), "MAM": (3, 4, 5)}
SECTORS = {
    "WS":  (-60.0,   20.0), "KH":  ( 20.0,   90.0), "EA":  ( 90.0,  160.0),
    "RS":  (160.0, -130.0), "ABS": (-130.0, -60.0),
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


def main():
    ds = xr.open_dataset(IN_NC, chunks={"time": 500})
    lon = ds["lon"].load()
    da = ds["divergence"]

    rows = []
    for sec, (lo, hi) in SECTORS.items():
        mask = sector_mask(lon, lo, hi)
        n_total = int(mask.sum().item())
        valid = da.where(mask).notnull().sum(dim=["y", "x"])  # valid cells per day
        valid = valid.sel(time=valid["time"].dt.year >= START_YEAR)
        frac = (valid / n_total)

        df = frac.to_dataframe(name="valid_frac").reset_index()
        dt_index = pd.DatetimeIndex(df["time"])

        for season, months in SEASONS.items():
            sub = df[dt_index.month.isin(months)].copy()
            sub["syear"] = season_year(pd.DatetimeIndex(sub["time"]), season)
            yearly = sub.groupby("syear")["valid_frac"].mean().reset_index()
            pre = yearly.loc[yearly.syear < BREAK_YEAR, "valid_frac"]
            post = yearly.loc[yearly.syear >= BREAK_YEAR, "valid_frac"]
            rows.append(dict(
                sector=sec, season=season, n_total_cells=n_total,
                pre_mean_valid_frac=pre.mean(), post_mean_valid_frac=post.mean(),
                pct_change=(post.mean() / pre.mean() - 1) * 100 if pre.mean() else np.nan,
                n_years_pre=len(pre), n_years_post=len(post),
            ))

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False, float_format="%.4g")
    print(f"wrote {OUT_CSV}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
