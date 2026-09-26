"""
06i_beta_div_interior.py -- Ch4, EXPLORATORY. The deformation channel.

Does the coupling between wind-stress DIVERGENCE and ice DIVERGENCE over the
persistently ice-covered interior differ before vs after 2016?  This is the
interior counterpart of the paper's beta (wind -> area): same interaction model,
same year-block bootstrap, but y = interior-mean ice divergence anomaly and
x = interior-mean wind-stress divergence anomaly, on the 06f fixed cells.

    y'_t = b1 x'_t + b2 post_t + b3 (x'_t * post_t) + r_t

It also reports var(x') and var(y') pre/post and R^2 pre/post, so the change in
ice-divergence variability can be split into "forcing changed" (var x),
"transfer changed" (b), and "everything else" (residual).

WIND COMPONENT ALIGNMENT -- READ THIS BEFORE RUNNING
  MODE = "geo" : tau_x/tau_y are EASTWARD/NORTHWARD (ERA5 regridded as scalars).
                 They are rotated into grid-x/grid-y using the 2-D lat/lon before
                 differencing.  Safe default unless the pipeline rotated them.
  MODE = "grid": tau_x/tau_y are already aligned with the EASE grid axes.

Scale note: if the wind file carries the /86400-instead-of-/3600 bug, |tau| will
be ~24x too small (typical |tau| should be ~0.03-0.3 N m^-2).  That rescales b
but leaves the interaction p-value, variance ratios and R^2 unchanged.
"""
import numpy as np
import pandas as pd
import xarray as xr

ROOT = "/user/geog/falejandraperez/sea-ice-phase"
ICE_NC = f"{ROOT}/results/ch4/derived_nc/ease_divergence_with_latlon.nc"
WIND_NC = "/user/geog/falejandraperez/sea-ice-phase/results/ch4/derived_nc/wind_stress_on_ease_sh.nc"          # <-- fill in from `find`
MODE = "geo"                              # "geo" or "grid" -- see docstring
OUT_CSV = f"{ROOT}/results/ch4/tables/beta_div_interior_prepost.csv"

START_YEAR, END_YEAR = 1988, 2023
BREAK_YEAR = 2016
COVERAGE_THRESHOLD = 0.80
MIN_CELLS = 30
CLIM_HALFWIN = 15        # +/- days for the day-of-year climatology smoothing
N_BOOT = 1000
SEED = 7

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


def time_name(ds):
    for c in ("time", "valid_time"):
        if c in ds.dims:
            return c
    raise KeyError(f"no time dim in {list(ds.dims)}")


def grid_unit_vectors(lat2d, y, x):
    """Local north and east unit vectors in grid (x, y) coordinates, from lat/lon.
    North = direction of increasing latitude (toward the equator in the SH).
    East  = north rotated 90 deg clockwise (map viewed from above, y up, x right)."""
    dlat_dy, dlat_dx = np.gradient(lat2d, y, x)
    nrm = np.hypot(dlat_dx, dlat_dy)
    n_x, n_y = dlat_dx / nrm, dlat_dy / nrm          # north unit vector (x, y comps)
    e_x, e_y = n_y, -n_x                              # east = clockwise 90 deg from north
    return n_x, n_y, e_x, e_y


def wind_divergence(tx, ty, lat2d, y, x, mode):
    """tx, ty: (time, y, x) numpy arrays. Returns divergence (time, y, x) in tau units / m."""
    if mode == "geo":
        n_x, n_y, e_x, e_y = grid_unit_vectors(lat2d, y, x)
        gx = tx * e_x + ty * n_x
        gy = tx * e_y + ty * n_y
    elif mode == "grid":
        gx, gy = tx, ty
    else:
        raise ValueError(mode)
    dgx_dx = np.gradient(gx, x, axis=2)
    dgy_dy = np.gradient(gy, y, axis=1)
    return dgx_dx + dgy_dy


def doy_anomaly(s):
    """Remove a smoothed day-of-year climatology (circular +/- CLIM_HALFWIN days)."""
    s = s.dropna()
    doy = s.index.dayofyear.values
    clim = s.groupby(doy).mean()
    clim = clim.reindex(range(1, 367)).interpolate(limit_direction="both")
    padded = pd.concat([clim.iloc[-CLIM_HALFWIN:], clim, clim.iloc[:CLIM_HALFWIN]])
    sm = padded.rolling(2 * CLIM_HALFWIN + 1, center=True).mean().iloc[CLIM_HALFWIN:-CLIM_HALFWIN]
    sm.index = clim.index
    return s - sm.reindex(doy).values


def fit_interaction(x, y, post):
    X = np.column_stack([np.ones_like(x), x, post, x * post])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta  # [b0, b1, b2, b3]


def r2(x, y):
    if len(x) < 3 or np.var(x) == 0:
        return np.nan
    b = np.polyfit(x, y, 1)
    res = y - np.polyval(b, x)
    return 1 - np.var(res) / np.var(y)


def main():
    rng = np.random.default_rng(SEED)
    ice = xr.open_dataset(ICE_NC, chunks={"time": 500})
    wind = xr.open_dataset(WIND_NC, chunks={time_name(xr.open_dataset(WIND_NC)): 500})
    wt = time_name(wind)
    lat2d = ice["lat"].load().values
    lon2d = ice["lon"].load().values
    x = ice["x"].values.astype("float64")
    y = ice["y"].values.astype("float64")
    assert wind["tau_x"].shape[-2:] == ice["divergence"].shape[-2:], "wind/ice grids differ"

    # magnitude sanity check on the wind file
    probe = np.hypot(wind["tau_x"].isel({wt: slice(5000, 5010)}).values,
                     wind["tau_y"].isel({wt: slice(5000, 5010)}).values)
    print(f"median |tau| in probe: {np.nanmedian(probe):.4g}  (expect ~0.03-0.3 N m^-2; "
          f"~0.001-0.01 means the /86400 scale bug is still in this file)")

    # common daily calendar
    ice_t = pd.DatetimeIndex(ice["time"].values).normalize()
    wind_t = pd.DatetimeIndex(wind[wt].values).normalize()
    common = ice_t.intersection(wind_t)
    common = common[(common.year >= START_YEAR) & (common.year <= END_YEAR)]
    print(f"common days {START_YEAR}-{END_YEAR}: {len(common)}")

    rows = []
    for season, months in SEASONS.items():
        days = common[common.month.isin(months)]
        i_ice = np.searchsorted(ice_t, days)
        i_wind = np.searchsorted(wind_t, days)

        div_ice = ice["divergence"].isel(time=i_ice).values.astype("float32")
        tx = wind["tau_x"].isel({wt: i_wind}).values.astype("float32")
        ty = wind["tau_y"].isel({wt: i_wind}).values.astype("float32")
        div_wind = wind_divergence(tx, ty, lat2d, y, x, MODE).astype("float32")
        del tx, ty

        valid = np.isfinite(div_ice)
        yrs = days.year.values
        pre_frac = valid[yrs < BREAK_YEAR].mean(axis=0)
        post_frac = valid[yrs >= BREAK_YEAR].mean(axis=0)
        fixed_all = (pre_frac >= COVERAGE_THRESHOLD) & (post_frac >= COVERAGE_THRESHOLD)

        for sec, (lo, hi) in SECTORS.items():
            mask = fixed_all & sector_mask(lon2d, lo, hi)
            n_cells = int(mask.sum())
            row = dict(season=season, sector=sec, n_fixed_cells=n_cells)
            if n_cells < MIN_CELLS:
                row["note"] = "TOO FEW FIXED CELLS"
                rows.append(row)
                continue

            y_s = pd.Series(np.nanmean(div_ice[:, mask], axis=1), index=days)
            x_s = pd.Series(np.nanmean(div_wind[:, mask], axis=1), index=days)
            ya, xa = doy_anomaly(y_s), doy_anomaly(x_s)
            df = pd.concat([xa.rename("x"), ya.rename("y")], axis=1).dropna()
            df["year"] = df.index.year
            df["post"] = (df.year >= BREAK_YEAR).astype(float)

            b = fit_interaction(df.x.values, df.y.values, df.post.values)
            pre, post = df[df.post == 0], df[df.post == 1]

            # year-block bootstrap, stratified by period
            yrs_pre, yrs_post = pre.year.unique(), post.year.unique()
            g = {yr: d for yr, d in df.groupby("year")}
            boots = np.empty((N_BOOT, 4))
            for k in range(N_BOOT):
                sel = np.concatenate([rng.choice(yrs_pre, len(yrs_pre)),
                                      rng.choice(yrs_post, len(yrs_post))])
                bs = pd.concat([g[yr] for yr in sel])
                boots[k] = fit_interaction(bs.x.values, bs.y.values, bs.post.values)
            lo_, hi_ = np.percentile(boots, [2.5, 97.5], axis=0)
            p_b3 = 2 * min((boots[:, 3] > 0).mean(), (boots[:, 3] < 0).mean())

            row.update(
                n_days_pre=len(pre), n_days_post=len(post),
                beta_pre=b[1], beta_pre_lo=lo_[1], beta_pre_hi=hi_[1],
                beta_post=b[1] + b[3],
                beta_post_lo=np.percentile(boots[:, 1] + boots[:, 3], 2.5),
                beta_post_hi=np.percentile(boots[:, 1] + boots[:, 3], 97.5),
                beta_change_pct=(b[3] / b[1]) * 100 if b[1] != 0 else np.nan,
                b3_p_boot=p_b3,
                var_winddiv_ratio_post_pre=post.x.var() / pre.x.var(),
                var_icediv_ratio_post_pre=post.y.var() / pre.y.var(),
                r2_pre=r2(pre.x.values, pre.y.values),
                r2_post=r2(post.x.values, post.y.values),
                note="ok",
            )
            rows.append(row)
            print(f"{season} {sec}: n={n_cells} beta_pre={b[1]:.3g} beta_post={b[1]+b[3]:.3g} "
                  f"p(b3)={p_b3:.3f} var(x) post/pre={row['var_winddiv_ratio_post_pre']:.2f} "
                  f"var(y) post/pre={row['var_icediv_ratio_post_pre']:.2f} "
                  f"R2 pre/post={row['r2_pre']:.2f}/{row['r2_post']:.2f}")
        del div_ice, div_wind

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False, float_format="%.6g")
    print(f"\nwrote {OUT_CSV}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
