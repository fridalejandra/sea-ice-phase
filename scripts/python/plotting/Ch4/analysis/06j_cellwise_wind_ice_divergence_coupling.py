"""
06j_cellwise_wind_ice_divergence_coupling.py -- Ch4. Cell-wise coupling between
wind-stress divergence and ice divergence, pre vs post 2016, interior and edge.

For every grid cell (not sector means), regress the cell's ice-divergence
anomaly on the SAME cell's wind-stress-divergence anomaly, separately for the
pre-2016 and post-2016 periods, and report the spatial distribution of

    r      Pearson correlation (scale-free; comparable to de Jager & Vichi's
           atmosphere-ice vorticity coupling)
    beta   slope (ice divergence per unit wind-stress divergence)
    var    variance of the wind-divergence forcing and of the ice divergence

Zones (as in 06f/06g): interior = >=80% valid ice-divergence days in BOTH
periods; edge = fails that but >=10% coverage in at least one period.

Sector-level summaries are MEDIANS across the zone's cells. Their uncertainty
and the pre/post difference come from a year-block bootstrap (resample season
years within each period, recompute every cell, take the median). A per-cell
Welch t-test on season-year betas (pre vs post) with BH-FDR within each
sector-season gives the fraction of cells whose coupling changed.

Wind components: MODE "geo" rotates ERA5 east/north stress into grid x/y using
the 2-D lat/lon before differencing. As a built-in sanity check the script also
computes the "grid" (unrotated) divergence and prints the SH-wide median r for
both; the physically correct one should give the higher |r|.

Outputs
  tables/cellwise_coupling_by_sector_season.csv
  derived_nc/cellwise_coupling_{season}.nc   (per-cell maps of r, beta, var ratios)
"""
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

ROOT = "/user/geog/falejandraperez/sea-ice-phase"
ICE_NC = f"{ROOT}/results/ch4/derived_nc/ease_divergence_with_latlon.nc"
WIND_NC = f"{ROOT}/results/ch4/derived_nc/wind_stress_on_ease_sh.nc"
OUT_CSV = f"{ROOT}/results/ch4/tables/cellwise_coupling_by_sector_season.csv"
OUT_NC = f"{ROOT}/results/ch4/derived_nc/cellwise_coupling_{{season}}.nc"
MODE = "geo"

START_YEAR, END_YEAR = 1988, 2023
BREAK_YEAR = 2016
COVERAGE_THRESHOLD = 0.80
EDGE_MIN_COVERAGE = 0.10
MIN_DAYS_PERIOD = 120      # a cell needs this many valid days in a period to be scored
MIN_DAYS_YEAR = 20         # ... and this many in a season-year for a per-year beta
MIN_CELLS = 30
CLIM_HALFWIN = 15
N_BOOT = 500
FDR_Q = 0.05
SEED = 7

SEASONS = {"JJA": (6, 7, 8), "SON": (9, 10, 11)}
SECTORS = {
    "WS":  (-60.0,   20.0), "KH":  ( 20.0,   90.0), "EA":  ( 90.0,  160.0),
    "RS":  (160.0, -130.0), "ABS": (-130.0, -60.0),
}


# ----------------------------------------------------------------------------- helpers
def sector_mask(lon, lo, hi):
    lon = ((lon + 180) % 360) - 180
    if lo <= hi:
        return (lon >= lo) & (lon < hi)
    return (lon >= lo) | (lon < hi)


def time_name(ds):
    for c in ("time", "valid_time"):
        if c in ds.dims:
            return c
    raise KeyError(f"no time dim in {list(ds.dims)}")


def grid_unit_vectors(lat2d, y, x):
    dlat_dy, dlat_dx = np.gradient(lat2d, y, x)
    nrm = np.hypot(dlat_dx, dlat_dy)
    n_x, n_y = dlat_dx / nrm, dlat_dy / nrm
    e_x, e_y = n_y, -n_x
    return n_x, n_y, e_x, e_y


def wind_divergence(tx, ty, lat2d, y, x, mode):
    if mode == "geo":
        n_x, n_y, e_x, e_y = grid_unit_vectors(lat2d, y, x)
        gx = tx * e_x + ty * n_x
        gy = tx * e_y + ty * n_y
    else:
        gx, gy = tx, ty
    return np.gradient(gx, x, axis=2) + np.gradient(gy, y, axis=1)


def doy_anomaly_2d(arr, doy):
    """arr: (ndays, ncells) with NaNs; doy: (ndays,). Smoothed per-cell day-of-year
    climatology over the season's doys, then subtract."""
    doys = np.unique(doy)
    clim = np.full((doys.size, arr.shape[1]), np.nan, dtype="float64")
    for i, d in enumerate(doys):
        clim[i] = np.nanmean(arr[doy == d], axis=0)
    # nan-aware running mean along doy (+/- CLIM_HALFWIN), clipped at the season edges
    sm = np.full_like(clim, np.nan)
    for i in range(doys.size):
        lo, hi = max(0, i - CLIM_HALFWIN), min(doys.size, i + CLIM_HALFWIN + 1)
        sm[i] = np.nanmean(clim[lo:hi], axis=0)
    idx = np.searchsorted(doys, doy)
    return arr - sm[idx]


def cell_stats(x, y):
    """x, y: (ndays, ncells) anomalies with NaNs. Returns beta, r, var_x, var_y, n per cell."""
    ok = np.isfinite(x) & np.isfinite(y)
    n = ok.sum(axis=0)
    xm = np.where(ok, x, 0.0); ym = np.where(ok, y, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mx = xm.sum(0) / n; my = ym.sum(0) / n
        sxx = ((xm - mx) ** 2 * ok).sum(0) / n
        syy = ((ym - my) ** 2 * ok).sum(0) / n
        sxy = ((xm - mx) * (ym - my) * ok).sum(0) / n
        beta = sxy / sxx
        r = sxy / np.sqrt(sxx * syy)
    bad = n < 3
    for a in (beta, r, sxx, syy):
        a[bad] = np.nan
    return beta, r, sxx, syy, n


def bh_fdr(p, q=FDR_Q):
    p = np.asarray(p, float); out = np.zeros(p.shape, bool)
    ok = np.isfinite(p); m = ok.sum()
    if m == 0:
        return out
    ps = np.sort(p[ok]); crit = q * np.arange(1, m + 1) / m
    passed = ps <= crit
    if passed.any():
        out[ok] = p[ok] <= ps[np.nonzero(passed)[0].max()]
    return out


# ----------------------------------------------------------------------------- main
def main():
    rng = np.random.default_rng(SEED)
    ice = xr.open_dataset(ICE_NC, chunks={"time": 500})
    wind = xr.open_dataset(WIND_NC)
    wt = time_name(wind)
    wind = wind.chunk({wt: 500})
    lat2d = ice["lat"].load().values; lon2d = ice["lon"].load().values
    x = ice["x"].values.astype("float64"); y = ice["y"].values.astype("float64")
    assert wind["tau_x"].shape[-2:] == ice["divergence"].shape[-2:], "wind/ice grids differ"

    probe = np.hypot(wind["tau_x"].isel({wt: slice(5000, 5010)}).values,
                     wind["tau_y"].isel({wt: slice(5000, 5010)}).values)
    print(f"median |tau| in probe: {np.nanmedian(probe):.4g}  "
          f"(expect ~0.03-0.3 N m^-2; ~0.001-0.01 = /86400 scale bug still present; "
          f"r is unaffected either way)")

    ice_t = pd.DatetimeIndex(ice["time"].values).normalize()
    wind_t = pd.DatetimeIndex(wind[wt].values).normalize()
    common = ice_t.intersection(wind_t)
    common = common[(common.year >= START_YEAR) & (common.year <= END_YEAR)]
    print(f"common days {START_YEAR}-{END_YEAR}: {len(common)}")

    rows = []
    for season, months in SEASONS.items():
        days = common[common.month.isin(months)]
        yrs = days.year.values; doy = days.dayofyear.values
        pre_d = yrs < BREAK_YEAR; post_d = ~pre_d
        i_ice = np.searchsorted(ice_t, days); i_wind = np.searchsorted(wind_t, days)

        div_ice = ice["divergence"].isel(time=i_ice).values.astype("float32")
        tx = wind["tau_x"].isel({wt: i_wind}).values.astype("float32")
        ty = wind["tau_y"].isel({wt: i_wind}).values.astype("float32")
        dw_geo = wind_divergence(tx, ty, lat2d, y, x, "geo").astype("float32")
        dw_grid = wind_divergence(tx, ty, lat2d, y, x, "grid").astype("float32")
        del tx, ty

        valid = np.isfinite(div_ice)
        pre_frac = valid[pre_d].mean(0); post_frac = valid[post_d].mean(0)
        interior = (pre_frac >= COVERAGE_THRESHOLD) & (post_frac >= COVERAGE_THRESHOLD)
        edge = ((pre_frac >= EDGE_MIN_COVERAGE) | (post_frac >= EDGE_MIN_COVERAGE)) & ~interior
        any_cells = interior | edge

        # flatten to (ndays, ncells) over cells we might use
        Y = doy_anomaly_2d(div_ice[:, any_cells].astype("float64"), doy)
        Xg = doy_anomaly_2d(dw_geo[:, any_cells].astype("float64"), doy)
        Xr = doy_anomaly_2d(dw_grid[:, any_cells].astype("float64"), doy)
        del div_ice, dw_geo, dw_grid

        # rotation sanity check: SH-wide, interior cells, all years
        int_cols = interior[any_cells]
        _, r_geo, _, _, _ = cell_stats(Xg[:, int_cols], Y[:, int_cols])
        _, r_grd, _, _, _ = cell_stats(Xr[:, int_cols], Y[:, int_cols])
        print(f"[{season}] rotation check, SH interior median r: geo={np.nanmedian(r_geo):.3f} "
              f"grid={np.nanmedian(r_grd):.3f}  (the physically correct mode should be larger)")
        X = Xg if MODE == "geo" else Xr
        del Xr

        # per-cell maps (full grid) to save
        maps = {k: np.full(lat2d.shape, np.nan, "float32") for k in
                ("r_pre", "r_post", "beta_pre", "beta_post", "var_wind_ratio", "var_ice_ratio",
                 "beta_change_p", "zone")}
        cell_index = np.flatnonzero(any_cells)

        for zone_name, zone_mask in (("interior", interior), ("edge", edge)):
            zcols = zone_mask[any_cells]
            for sec, (lo, hi) in SECTORS.items():
                cols = zcols & sector_mask(lon2d, lo, hi)[any_cells]
                if cols.sum() < MIN_CELLS:
                    rows.append(dict(season=season, zone=zone_name, sector=sec,
                                     n_cells=int(cols.sum()), note="TOO FEW CELLS"))
                    continue
                xs, ys = X[:, cols], Y[:, cols]

                b_pre, r_pre, vx_pre, vy_pre, n_pre = cell_stats(xs[pre_d], ys[pre_d])
                b_post, r_post, vx_post, vy_post, n_post = cell_stats(xs[post_d], ys[post_d])
                scored = (n_pre >= MIN_DAYS_PERIOD) & (n_post >= MIN_DAYS_PERIOD)
                if scored.sum() < MIN_CELLS:
                    rows.append(dict(season=season, zone=zone_name, sector=sec,
                                     n_cells=int(scored.sum()), note="TOO FEW SCORED CELLS"))
                    continue
                for a in (b_pre, r_pre, vx_pre, vy_pre, b_post, r_post, vx_post, vy_post):
                    a[~scored] = np.nan

                # per-cell change test: Welch t on season-year betas
                by = {}
                for yr in np.unique(yrs):
                    sel = yrs == yr
                    b_y, _, _, _, n_y = cell_stats(xs[sel], ys[sel])
                    b_y[n_y < MIN_DAYS_YEAR] = np.nan
                    by[yr] = b_y
                B = np.array([by[yr] for yr in sorted(by)])          # (nyears, ncells)
                Byrs = np.array(sorted(by))
                Bpre, Bpost = B[Byrs < BREAK_YEAR], B[Byrs >= BREAK_YEAR]
                with np.errstate(invalid="ignore"):
                    t, p = stats.ttest_ind(Bpre, Bpost, equal_var=False, nan_policy="omit")
                p = np.asarray(p, float); p[~scored] = np.nan
                sig = bh_fdr(p)
                dec = sig & (np.nanmedian(Bpost, 0) < np.nanmedian(Bpre, 0))
                inc = sig & ~dec

                # year-block bootstrap of the sector MEDIANS (r and beta), pre and post
                yrs_pre = np.unique(yrs[pre_d]); yrs_post = np.unique(yrs[post_d])
                day_idx_by_year = {yr: np.flatnonzero(yrs == yr) for yr in np.unique(yrs)}
                boot = np.empty((N_BOOT, 4))   # med_r_pre, med_r_post, med_b_pre, med_b_post
                for k in range(N_BOOT):
                    ip = np.concatenate([day_idx_by_year[yr] for yr in rng.choice(yrs_pre, yrs_pre.size)])
                    io = np.concatenate([day_idx_by_year[yr] for yr in rng.choice(yrs_post, yrs_post.size)])
                    bp, rp, *_ = cell_stats(xs[ip], ys[ip]); bo, ro, *_ = cell_stats(xs[io], ys[io])
                    bp[~scored] = rp[~scored] = bo[~scored] = ro[~scored] = np.nan
                    boot[k] = (np.nanmedian(rp), np.nanmedian(ro), np.nanmedian(bp), np.nanmedian(bo))
                d_r = boot[:, 1] - boot[:, 0]; d_b = boot[:, 3] - boot[:, 2]
                p_r = 2 * min((d_r > 0).mean(), (d_r < 0).mean())
                p_b = 2 * min((d_b > 0).mean(), (d_b < 0).mean())
                ci = lambda v: tuple(np.percentile(v, [2.5, 97.5]))

                rows.append(dict(
                    season=season, zone=zone_name, sector=sec, n_cells=int(scored.sum()),
                    median_r_pre=np.nanmedian(r_pre), median_r_post=np.nanmedian(r_post),
                    r_pre_ci_lo=ci(boot[:, 0])[0], r_pre_ci_hi=ci(boot[:, 0])[1],
                    r_post_ci_lo=ci(boot[:, 1])[0], r_post_ci_hi=ci(boot[:, 1])[1],
                    delta_median_r=np.nanmedian(r_post) - np.nanmedian(r_pre), p_delta_r=p_r,
                    median_R2_pre=np.nanmedian(r_pre ** 2), median_R2_post=np.nanmedian(r_post ** 2),
                    median_beta_pre=np.nanmedian(b_pre), median_beta_post=np.nanmedian(b_post),
                    delta_median_beta_pct=(np.nanmedian(b_post) / np.nanmedian(b_pre) - 1) * 100,
                    p_delta_beta=p_b,
                    frac_cells_beta_sig_decrease=dec.sum() / scored.sum(),
                    frac_cells_beta_sig_increase=inc.sum() / scored.sum(),
                    median_var_winddiv_ratio=np.nanmedian(vx_post / vx_pre),
                    median_var_icediv_ratio=np.nanmedian(vy_post / vy_pre),
                    note="ok"))
                print(f"{season} {zone_name:8s} {sec:3s} n={scored.sum():5d}  "
                      f"r {np.nanmedian(r_pre):.3f}->{np.nanmedian(r_post):.3f} (p={p_r:.3f})  "
                      f"beta {np.nanmedian(b_pre):.3g}->{np.nanmedian(b_post):.3g} (p={p_b:.3f})  "
                      f"var wind {np.nanmedian(vx_post / vx_pre):.2f}  var ice {np.nanmedian(vy_post / vy_pre):.2f}  "
                      f"cells beta dec/inc {dec.sum() / scored.sum():.2f}/{inc.sum() / scored.sum():.2f}")

                # store maps
                gi = cell_index[cols]
                for key, arr in (("r_pre", r_pre), ("r_post", r_post), ("beta_pre", b_pre),
                                 ("beta_post", b_post), ("var_wind_ratio", vx_post / vx_pre),
                                 ("var_ice_ratio", vy_post / vy_pre), ("beta_change_p", p)):
                    maps[key].ravel()[gi] = arr
                maps["zone"].ravel()[gi] = 1.0 if zone_name == "interior" else 2.0

        ds_out = xr.Dataset({k: (("y", "x"), v) for k, v in maps.items()},
                            coords={"y": y, "x": x, "lat": (("y", "x"), lat2d), "lon": (("y", "x"), lon2d)})
        ds_out.attrs.update(season=season, mode=MODE, break_year=BREAK_YEAR,
                            zone_codes="1=interior, 2=edge", note="pre/post per-cell coupling of ice "
                            "divergence anomaly on wind-stress divergence anomaly")
        ds_out.to_netcdf(OUT_NC.format(season=season))
        print(f"wrote {OUT_NC.format(season=season)}")
        del X, Y

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False, float_format="%.5g")
    print(f"\nwrote {OUT_CSV}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
