"""
06h_divergence_interior_vs_edge_interaction.py -- Ch4, EXPLORATORY (not for the
dissertation). Does the interior's divergence/convergence-variability trend
differ from the edge's -- tested directly as ONE interaction term, rather than
by comparing 06f's and 06g's regressions side by side.

For each variable x sector x season, pools the interior (fixed-cell, 06f
definition: >=80% coverage in both periods) and edge (06g definition: fails
that, but >=10% coverage in at least one period) season-year log2-variance
series into one table with a zone indicator, and fits

    log2var ~ year_c * zone

zone is categorical with "interior" as the reference level, so the
year_c:zone[T.edge] coefficient estimates how much the edge's trend differs
from the interior's trend, with its own p-value -- a single, direct test of
the redistribution hypothesis. BH-FDR (q=0.05) is applied across all fitted
rows at the end.
"""
import numpy as np
import pandas as pd
import xarray as xr
import statsmodels.formula.api as smf

ROOT = "/user/geog/falejandraperez/sea-ice-phase"
IN_NC = f"{ROOT}/results/ch4/derived_nc/ease_divergence_with_latlon.nc"
OUT_CSV = f"{ROOT}/results/ch4/tables/divergence_interior_vs_edge_interaction.csv"

START_YEAR = 1988
BREAK_YEAR = 2016
COVERAGE_THRESHOLD = 0.80
EDGE_MIN_COVERAGE = 0.10
MIN_DAYS = 20
MIN_SEASON_YEARS = 10
MIN_POST_YEARS = 4
MIN_CELLS = 30
FDR_Q = 0.05

VARS = ["divergence", "div_positive", "div_negative"]
SEASONS = {"JJA": (6, 7, 8), "SON": (9, 10, 11)}
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


def season_year_variances(da_season, mask, season):
    daily = da_season.where(mask).mean(dim=["y", "x"], skipna=True)
    df = daily.to_dataframe(name="value").reset_index().dropna(subset=["value"])
    dt_index = pd.DatetimeIndex(df["time"])
    df["syear"] = season_year(dt_index, season)
    rows = []
    for yr, g in df.groupby("syear"):
        if len(g) < MIN_DAYS:
            continue
        v = g["value"].var(ddof=1)
        if pd.notna(v) and v > 0:
            rows.append((yr, v))
    return pd.DataFrame(rows, columns=["year", "var"])


def bh_fdr(pvals, q=FDR_Q):
    p = np.asarray(pvals, dtype=float)
    ok = np.isfinite(p)
    out = np.zeros(p.shape, dtype=bool)
    m = ok.sum()
    if m == 0:
        return out
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    ranked = p[order]
    thresh_line = q * (np.arange(1, m + 1) / m)
    passed = ranked <= thresh_line
    if passed.any():
        cutoff = ranked[np.nonzero(passed)[0].max()]
        out[ok] = p[ok] <= cutoff
    return out


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

                fixed_mask = smask & (pre_frac >= COVERAGE_THRESHOLD) & (post_frac >= COVERAGE_THRESHOLD)
                has_presence = (pre_frac >= EDGE_MIN_COVERAGE) | (post_frac >= EDGE_MIN_COVERAGE)
                edge_mask = smask & has_presence & (~fixed_mask)

                n_interior = int(fixed_mask.sum().item())
                n_edge = int(edge_mask.sum().item())

                vdf_int = season_year_variances(da_season, fixed_mask, season)
                vdf_edge = season_year_variances(da_season, edge_mask, season)

                result = dict(variable=var, sector=sec, season=season,
                              n_interior_cells=n_interior, n_edge_cells=n_edge,
                              n_years_interior=len(vdf_int), n_years_edge=len(vdf_edge))

                n_post_int = int((vdf_int.year >= BREAK_YEAR).sum()) if len(vdf_int) else 0
                n_post_edge = int((vdf_edge.year >= BREAK_YEAR).sum()) if len(vdf_edge) else 0
                enough = (n_interior >= MIN_CELLS and n_edge >= MIN_CELLS and
                          len(vdf_int) >= MIN_SEASON_YEARS and len(vdf_edge) >= MIN_SEASON_YEARS and
                          n_post_int >= MIN_POST_YEARS and n_post_edge >= MIN_POST_YEARS)

                if not enough:
                    result["note"] = "insufficient data"
                    rows.append(result)
                    continue

                ref_int = vdf_int.loc[vdf_int.year < BREAK_YEAR, "var"].mean()
                ref_edge = vdf_edge.loc[vdf_edge.year < BREAK_YEAR, "var"].mean()
                vdf_int = vdf_int.assign(zone="interior", log2var=np.log2(vdf_int["var"] / ref_int))
                vdf_edge = vdf_edge.assign(zone="edge", log2var=np.log2(vdf_edge["var"] / ref_edge))
                both = pd.concat([vdf_int, vdf_edge], ignore_index=True)
                both["year_c"] = both["year"] - both["year"].mean()
                both["zone"] = pd.Categorical(both["zone"], categories=["interior", "edge"])

                m = smf.ols("log2var ~ year_c * zone", data=both).fit()
                inter_term = "year_c:zone[T.edge]"
                beta_int = m.params["year_c"]
                beta_edge = beta_int + m.params[inter_term]

                result["interior_trend_pct_per_decade"] = (2 ** (beta_int * 10) - 1) * 100
                result["edge_trend_pct_per_decade"] = (2 ** (beta_edge * 10) - 1) * 100
                result["interaction_p"] = m.pvalues[inter_term]
                result["note"] = "ok"
                rows.append(result)

    out = pd.DataFrame(rows)
    fitted = out["note"] == "ok"
    out["interaction_sig_fdr"] = False
    out.loc[fitted, "interaction_sig_fdr"] = bh_fdr(out.loc[fitted, "interaction_p"].values)

    out.to_csv(OUT_CSV, index=False, float_format="%.6g")
    print(f"wrote {OUT_CSV}: {len(out)} rows, {fitted.sum()} fitted")
    print(out.to_string(index=False))
    n_sig = int(out["interaction_sig_fdr"].sum())
    print(f"\n{n_sig} of {int(fitted.sum())} fitted interaction terms significant after BH-FDR (q={FDR_Q})")


if __name__ == "__main__":
    main()
