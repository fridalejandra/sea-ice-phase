"""
06l_coupling_scale_diagnostic.py -- why is cell-wise wind-divergence / ice-divergence
coupling ~0?  Ladder of checks on a 5-winter subsample (fast).

  A. exact-zero fraction in ice divergence (fill values masquerading as data)
  B. lag sweep (-2..+2 days) at one box size  -> day-convention offset?
  C. box-average sweep 1,3,5,9,15 cells (25..375 km) at lag 0 -> scale of coupling
  D. (optional) ice VELOCITY vs wind stress, cell-wise -> ground truth that the
     motion product and wind are paired correctly.  Set UV_NC/U_VAR/V_VAR below.

Median Pearson r across interior cells is reported for each case.
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import uniform_filter

ROOT = "/user/geog/falejandraperez/sea-ice-phase"
ICE_NC = f"{ROOT}/results/ch4/derived_nc/ease_divergence_with_latlon.nc"
WIND_NC = f"{ROOT}/results/ch4/derived_nc/wind_stress_on_ease_sh.nc"
UV_NC = ""            # <-- optional: NSIDC u,v on the same EASE grid; leave "" to skip D
U_VAR, V_VAR = "u", "v"
UV_TIME = None        # name of the time dim in UV_NC; None = auto

YEARS = range(2008, 2015)
MONTHS = (6, 7, 8)
BOXES = (1, 3, 5, 9, 15)
LAGS = (-2, -1, 0, 1, 2)
LAG_BOX = 5
COVERAGE = 0.80
CLIM_HALFWIN = 15


def time_name(ds):
    for c in ("time", "valid_time"):
        if c in ds.dims:
            return c
    raise KeyError(list(ds.dims))


def grid_unit_vectors(lat2d, y, x):
    dlat_dy, dlat_dx = np.gradient(lat2d, y, x)
    nrm = np.hypot(dlat_dx, dlat_dy)
    with np.errstate(invalid="ignore"):
        n_x, n_y = dlat_dx / nrm, dlat_dy / nrm
    return n_x, n_y, n_y, -n_x          # n_x, n_y, e_x, e_y


def to_grid(tx, ty, lat2d, y, x):
    n_x, n_y, e_x, e_y = grid_unit_vectors(lat2d, y, x)
    return tx * e_x + ty * n_x, tx * e_y + ty * n_y


def divergence(gx, gy, y, x):
    return np.gradient(gx, x, axis=2) + np.gradient(gy, y, axis=1)


def box_mean(a, k):
    """nan-aware (time, y, x) box mean of size k."""
    if k == 1:
        return a
    ok = np.isfinite(a).astype("float32")
    num = uniform_filter(np.where(np.isfinite(a), a, 0).astype("float32"), size=(1, k, k), mode="constant")
    den = uniform_filter(ok, size=(1, k, k), mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den < 0.5] = np.nan          # require at least half the box valid
    return out


def doy_anom(a, doy):
    """per-cell smoothed day-of-year climatology removed; a: (t, y, x)."""
    doys = np.unique(doy)
    clim = np.stack([np.nanmean(a[doy == d], axis=0) for d in doys])
    sm = np.stack([np.nanmean(clim[max(0, i - CLIM_HALFWIN):i + CLIM_HALFWIN + 1], axis=0)
                   for i in range(doys.size)])
    return a - sm[np.searchsorted(doys, doy)]


def median_r(a, b, mask):
    """median over masked cells of the temporal Pearson r between a and b (t, y, x)."""
    A, B = a[:, mask], b[:, mask]
    ok = np.isfinite(A) & np.isfinite(B)
    n = ok.sum(0)
    A = np.where(ok, A, 0.0); B = np.where(ok, B, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ma, mb = A.sum(0) / n, B.sum(0) / n
        saa = (((A - ma) ** 2) * ok).sum(0); sbb = (((B - mb) ** 2) * ok).sum(0)
        sab = ((A - ma) * (B - mb) * ok).sum(0)
        r = sab / np.sqrt(saa * sbb)
    r[n < 30] = np.nan
    return float(np.nanmedian(r)), int(np.isfinite(r).sum())


def main():
    ice = xr.open_dataset(ICE_NC); wind = xr.open_dataset(WIND_NC); wt = time_name(wind)
    lat2d = ice["lat"].values; x = ice["x"].values.astype(float); y = ice["y"].values.astype(float)
    it = pd.DatetimeIndex(ice["time"].values).normalize(); wtime = pd.DatetimeIndex(wind[wt].values).normalize()
    days = it.intersection(wtime); days = days[days.year.isin(YEARS) & days.month.isin(MONTHS)]
    doy = days.dayofyear.values
    print(f"subsample: {len(days)} days, {YEARS.start}-{YEARS.stop - 1} months {MONTHS}")

    div_ice = ice["divergence"].isel(time=np.searchsorted(it, days)).values.astype("float64")
    tx = wind["tau_x"].isel({wt: np.searchsorted(wtime, days)}).values.astype("float64")
    ty = wind["tau_y"].isel({wt: np.searchsorted(wtime, days)}).values.astype("float64")
    gx, gy = to_grid(tx, ty, lat2d, y, x)
    div_wind = divergence(gx, gy, y, x)

    interior = np.isfinite(div_ice).mean(0) >= COVERAGE
    print(f"interior cells (>= {COVERAGE:.0%} coverage in subsample): {interior.sum()}")

    # A. exact zeros
    fin = div_ice[np.isfinite(div_ice)]
    print(f"\nA. ice divergence: {np.mean(fin == 0):.2%} of finite values are EXACTLY zero "
          f"(should be ~0%; a large number means fill values are in the data)")
    print(f"   ice div  std {np.nanstd(div_ice):.3g} s^-1   wind div std {np.nanstd(div_wind):.3g} (tau units / m)")

    Yi = doy_anom(div_ice, doy); Xw = doy_anom(div_wind, doy)

    # C. box sweep, lag 0
    print("\nC. box-average sweep (lag 0), median r over interior cells:")
    for k in BOXES:
        r, n = median_r(box_mean(Xw, k), box_mean(Yi, k), interior)
        print(f"   box {k:2d} cells (~{25 * k:3d} km): median r = {r:+.3f}   (n cells = {n})")

    # B. lag sweep at LAG_BOX
    print(f"\nB. lag sweep at box {LAG_BOX} (positive lag = ice lags wind):")
    Xb, Yb = box_mean(Xw, LAG_BOX), box_mean(Yi, LAG_BOX)
    for L in LAGS:
        if L >= 0:
            r, n = median_r(Xb[:len(days) - L], Yb[L:], interior)
        else:
            r, n = median_r(Xb[-L:], Yb[:len(days) + L], interior)
        print(f"   lag {L:+d} d: median r = {r:+.3f}")

    # D. velocity vs stress
    if UV_NC and os.path.exists(UV_NC):
        uv = xr.open_dataset(UV_NC); ut = UV_TIME or time_name(uv)
        utime = pd.DatetimeIndex(uv[ut].values).normalize()
        d2 = days.intersection(utime)
        u = uv[U_VAR].isel({ut: np.searchsorted(utime, d2)}).values.astype("float64")
        v = uv[V_VAR].isel({ut: np.searchsorted(utime, d2)}).values.astype("float64")
        sel = np.searchsorted(days, d2)
        doy2 = d2.dayofyear.values
        ua, va = doy_anom(u, doy2), doy_anom(v, doy2)
        gxa, gya = doy_anom(gx[sel], doy2), doy_anom(gy[sel], doy2)
        m2 = np.isfinite(u).mean(0) >= COVERAGE
        print(f"\nD. ice velocity vs wind stress ({len(d2)} days, {m2.sum()} cells):")
        for k in (1, 5):
            ru, _ = median_r(box_mean(gxa, k), box_mean(ua, k), m2)
            rv, _ = median_r(box_mean(gya, k), box_mean(va, k), m2)
            print(f"   box {k}: median r(u, tau_x_grid) = {ru:+.3f}   r(v, tau_y_grid) = {rv:+.3f}   "
                  f"(literature: ~0.6-0.8)")
    else:
        print("\nD. skipped (set UV_NC to the NSIDC u,v file on the EASE grid to run it)")


if __name__ == "__main__":
    main()
