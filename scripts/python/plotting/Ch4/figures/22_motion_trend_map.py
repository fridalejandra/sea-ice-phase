#!/usr/bin/env python
"""
22_motion_trend_map.py -- Antarctic sea-ice motion trend map in the style of Webster et al. (2026),
Nat. Rev. Earth Environ., Fig. 5b (NSIDC-0116 v4, September-October, 1982-2024):
  shading = trend in mean drift speed (cm s^-1 per decade)
  arrows  = trend in the mean velocity vector (cm s^-1 per decade); black where the trend in
            either component is significant (p < 0.05), light grey otherwise.
Also prints the pan-Antarctic speed trend, to compare with their +0.69 cm s^-1 per decade
(a reproduction check of the pipeline), and the same for 1988-2024 (SSM/I era).

Drawn in the EASE-Grid 2.0 South projection so u, v (along grid x, y) need no rotation.
"""
import glob
import warnings
import os
import re
import sys
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ch4_style as st  # noqa: E402

ROOT = "/user/geog/falejandraperez/sea-ice-phase"
DRIFT_DIR = f"{ROOT}/data/drift_nsidc/"
DRIFT_GLOB = "icemotion_daily_sh_25km_*_v4.1.nc"
FIG = f"{ROOT}/results/ch4/figures"
OUT_NC = f"{ROOT}/results/ch4/derived_nc/motion_trend_{{season}}_{{y0}}.nc"
SEASONS = {"SO": (9, 10), "JJA": (6, 7, 8)}
START_YEARS = (1988,)
END = 2024
MIN_DAYS_FRAC = 0.5      # cell-season valid if drift on >= 50% of days
MIN_YEARS = 25
STRIDE = 7               # arrows every 7 cells (~175 km)
LAND = "0.88"            # house style; use "black" to match Webster et al. exactly
warnings.filterwarnings("ignore", category=RuntimeWarning)
EASE = ccrs.LambertAzimuthalEqualArea(central_longitude=0, central_latitude=-90,
                                      globe=ccrs.Globe(ellipse="WGS84"))


def drift_file(year):
    hits = [f for f in glob.glob(os.path.join(DRIFT_DIR, DRIFT_GLOB))
            if re.search(rf"_{year}\d{{4}}", os.path.basename(f))]
    return sorted(hits)[0] if hits else None


def times(ds, year):
    try:
        return pd.DatetimeIndex(ds.indexes["time"].to_datetimeindex())
    except Exception:
        try:
            return pd.DatetimeIndex(pd.to_datetime([str(t)[:10] for t in ds.time.values]))
        except Exception:
            return pd.date_range(f"{year}-01-01", periods=ds.sizes["time"], freq="D")


def seasonal_means(months, y0):
    yrs = np.arange(y0, END + 1)
    U = V = S = None
    xy = None
    for i, y in enumerate(yrs):
        f = drift_file(y)
        if f is None:
            continue
        ds = xr.open_dataset(f)
        t = times(ds, y)
        j = np.nonzero(np.isin(t.month, months))[0]
        u = ds["u"].isel(time=j).values.astype("float32")
        v = ds["v"].isel(time=j).values.astype("float32")
        if U is None:
            shp = (len(yrs),) + u.shape[1:]
            U, V, S = (np.full(shp, np.nan, "float32") for _ in range(3))
            xy = (ds["x"].values, ds["y"].values)
            lat = ds["latitude"].values if "latitude" in ds else None
        ok = np.isfinite(u) & np.isfinite(v)
        frac = ok.mean(0)
        good = frac >= MIN_DAYS_FRAC
        U[i] = np.where(good, np.nanmean(u, 0), np.nan)
        V[i] = np.where(good, np.nanmean(v, 0), np.nan)
        S[i] = np.where(good, np.nanmean(np.hypot(u, v), 0), np.nan)
        print(f"  {y}", flush=True)
    return yrs, U, V, S, xy


def cell_trend(Z, yrs):
    """OLS slope per cell (per decade) and p-value, NaN-aware."""
    ok = np.isfinite(Z)
    n = ok.sum(0)
    x = np.where(ok, (yrs[:, None, None] - yrs.mean()) / 10.0, np.nan)
    xm, zm = np.nanmean(x, 0), np.nanmean(Z, 0)
    sxx = np.nansum((x - xm) ** 2, 0)
    b = np.nansum((x - xm) * (Z - zm), 0) / sxx
    res = Z - (zm + b * (x - xm))
    se = np.sqrt(np.nansum(res ** 2, 0) / (n - 2) / sxx)
    p = 2 * stats.t.sf(np.abs(b / se), n - 2)
    bad = n < MIN_YEARS
    return np.where(bad, np.nan, b), np.where(bad, np.nan, p)


def plot(xy, bS, bU, bV, sig, title, out):
    x, y = xy
    X, Y = np.meshgrid(x, y)
    fig = plt.figure(figsize=(5.4, 5.6))
    ax = fig.add_subplot(1, 1, 1, projection=EASE)
    ax.set_extent([-4.0e6, 4.0e6, -4.0e6, 4.0e6], crs=EASE)
    im = ax.pcolormesh(x, y, bS, cmap="RdBu_r", vmin=-1.5, vmax=1.5, shading="auto", transform=EASE, zorder=1)
    k = (slice(None, None, STRIDE), slice(None, None, STRIDE))
    m = np.isfinite(bU[k]) & np.isfinite(bV[k])
    for mask, col in ((m & ~sig[k], "0.75"), (m & sig[k], "k")):
        q = ax.quiver(X[k][mask], Y[k][mask], bU[k][mask], bV[k][mask], transform=EASE, color=col,
                      scale=40, width=0.0032, headwidth=4, zorder=2)
    ax.quiverkey(q, 0.86, 0.03, 1, "1 cm s⁻¹ per decade", labelpos="W", fontproperties={"size": 7}, coordinates="axes")
    ax.add_feature(cfeature.LAND, facecolor=LAND, zorder=3)
    ax.coastlines(lw=0.3, color=st.INK, zorder=4)
    for sec, (lo, hi) in {"WS": (300, 20), "KH": (20, 90), "EA": (90, 160), "RA": (160, 230), "ABS": (230, 300)}.items():
        ax.plot([lo, lo], [-90, -57], color="0.55", lw=0.6, ls=(0, (3, 2)), transform=ccrs.PlateCarree(), zorder=5)
        mid = (lo + ((hi - lo) % 360) / 2) % 360
        ax.text(mid, -55.5, sec, color=st.SECTOR_COLORS[sec], ha="center", va="center",
                fontproperties=st.bold_font_properties(size=8), transform=ccrs.PlateCarree(), zorder=6)
    ax.spines["geo"].set_edgecolor("0.85")
    ax.set_title(title, loc="left")
    cb = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.04, pad=0.02, extend="both")
    cb.set_label("Trend in drift speed (cm s⁻¹ per decade)", fontsize=8)
    cb.outline.set_visible(False)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    for season, months in SEASONS.items():
        for y0 in START_YEARS:
            yrs, U, V, S, xy = seasonal_means(months, y0)
            bS, pS = cell_trend(S, yrs)
            bU, pU = cell_trend(U, yrs)
            bV, pV = cell_trend(V, yrs)
            sig = (pU < 0.05) | (pV < 0.05)
            # pan-Antarctic: mean speed over all valid cells each year, then trend
            pan = np.array([np.nanmean(s) for s in S])
            ok = np.isfinite(pan)
            lr = stats.linregress(yrs[ok], pan[ok])
            print(f"{season} {y0}-{END}: pan-Antarctic speed trend {lr.slope * 10:+.2f} cm/s per decade "
                  f"(p={lr.pvalue:.3f}); cells with significant speed trend: "
                  f"{np.nanmean(pS[np.isfinite(pS)] < 0.05):.0%}")
            xr.Dataset({"speed_trend": (("y", "x"), bS), "speed_p": (("y", "x"), pS),
                        "u_trend": (("y", "x"), bU), "v_trend": (("y", "x"), bV),
                        "vector_sig": (("y", "x"), sig.astype("int8"))},
                       coords={"x": xy[0], "y": xy[1]}).to_netcdf(OUT_NC.format(season=season, y0=y0))
            plot(xy, bS, bU, bV, sig, f"Antarctic sea-ice motion trend, {season} {y0}–{END}",
                 f"{FIG}/motion_trend_{season}_{y0}_{END}.png")


if __name__ == "__main__":
    main()
