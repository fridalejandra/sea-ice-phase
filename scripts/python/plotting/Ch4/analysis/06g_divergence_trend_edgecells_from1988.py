"""
06g_divergence_trend_edgecells_from1988.py -- Ch4: day-to-day
divergence/convergence variability trend, restricted to the EDGE/MIZ
cell set -- the complement of 06f's fixed-cell set. Where 06f keeps only
cells with >=80% valid-day coverage in BOTH the pre-2016 and post-2016
periods (reliable interior ice), this script keeps cells that FAIL that
fixed criterion but still have >=10% coverage in at least one period --
i.e. cells with real but variable ice presence, not permanently-open-water
cells that would contribute no signal.

This is the direct complement test for the "deformation relocated to the
edge, not just declined" hypothesis: if 06f shows declining divergence
variance in the interior while this script shows flat-or-increasing
variance at the edge, that's a falsifiable confirmation of redistribution
rather than a uniform quieting of the whole pack.

JJA/SON only, matching 06b/06f -- DJF/MAM have too little ice cover for a
meaningful cell set either way.
"""
import numpy as np
import pandas as pd
import xarray as xr
import statsmodels.formula.api as smf

ROOT = "/user/geog/falejandraperez/sea-ice-phase"
IN_NC = f"{ROOT}/results/ch4/derived_nc/ease_divergence_with_latlon.nc"
OUT_CSV = f"{ROOT}/results/ch4/tables/divergence_variance_trend_edgecells_from1988.csv"

START_YEAR = 1988
BREAK_YEAR = 2016
COVERAGE_THRESHOLD = 0.80   # same fixed-cell definition as 06f, used here to EXCLUDE
EDGE_MIN_COVERAGE = 0.10    # floor: must have some real presence in at least one period
MIN_DAYS = 20
MIN_SEASON_YEARS = 10
MIN_POST_YEARS = 4
MIN_EDGE_CELLS = 30         # below this, don't trust the sector mean -- flagged, not dropped

VARS = ["divergence", "div_positive", "div_negative"]
SEASONS = {"JJA": (6, 7, 8), "SON": (9, 10, 11)}   # JJA/SON only -- see docstring
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
    trend_pct_per_decade = (2 ** (m1.params["year_c"] * 10) - 1) * 100
    trend_p = m1.pvalues["year_c"]

    return dict(best_model=best, dAIC_vs_next=dAIC, **{f"AIC_{k}": v for k, v in aics.items()},
                step_trendadj_pct=step_pct, step_trendadj_p=step_p,
                trend_pct_per_decade=trend_pct_per_decade, trend_p=trend_p)


def main():
    ds = xr.open_dataset(IN_NC, chunks={"time": 500})
    lon = ds["lon"].load()
    rows = []

    for var in VARS:
        da = ds[var]
        for sec, (lo, hi) in SECTORS.items():
            smask = sector_mask(lon, lo, hi)

            for season, months in SEASONS.items():
                da_season = da.sel(time=da["time"].dt.month.isin(months))
                da_season = da_season.sel(time=da_season["time"].dt.year >= START_YEAR)
                years = da_season["time"].dt.year.values
                is_pre = years < BREAK_YEAR
                is_post = years >= BREAK_YEAR

                valid = ds["divergence"].sel(time=da_season["time"]).notnull()
                pre_frac = valid.isel(time=is_pre).mean(dim="time").compute()
                post_frac = valid.isel(time=is_post).mean(dim="time").compute()

                fixed_mask = (pre_frac >= COVERAGE_THRESHOLD) & (post_frac >= COVERAGE_THRESHOLD)
                has_presence = (pre_frac >= EDGE_MIN_COVERAGE) | (post_frac >= EDGE_MIN_COVERAGE)
                edge_mask = smask & has_presence & (~fixed_mask)
                n_edge = int(edge_mask.sum().item())

                daily = da_season.where(edge_mask).mean(dim=["y", "x"], skipna=True)
                df = daily.to_dataframe(name="value").reset_index().dropna(subset=["value"])
                dt_index = pd.DatetimeIndex(df["time"])
                df["syear"] = season_year(dt_index, season)

                var_rows = []
                for yr, g in df.groupby("syear"):
                    if len(g) < MIN_DAYS:
                        continue
                    v = g["value"].var(ddof=1)
                    if pd.notna(v) and v > 0:
                        var_rows.append((yr, v))

                vdf = pd.DataFrame(var_rows, columns=["year", "var"])
                n_post = int((vdf.year >= BREAK_YEAR).sum()) if len(vdf) else 0
                result = dict(variable=var, sector=sec, season=season,
                              n_edge_cells=n_edge, n_years=len(vdf), n_post=n_post,
                              note="ok" if n_edge >= MIN_EDGE_CELLS else "TOO FEW EDGE CELLS")

                if n_edge < MIN_EDGE_CELLS or len(vdf) < MIN_SEASON_YEARS or n_post < MIN_POST_YEARS:
                    rows.append(result)
                    continue

                ref = vdf.loc[vdf.year < BREAK_YEAR, "var"].mean()
                vdf["log2var"] = np.log2(vdf["var"] / ref)
                result.update(fit_models(vdf))
                rows.append(result)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False, float_format="%.6g")
    print(f"wrote {OUT_CSV}: {len(out)} rows")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
