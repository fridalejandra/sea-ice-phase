"""
fig_wind_djf_mam.py
Wind stress magnitude difference, DJF (left) and MAM (right), side by side.
DJF +11.4%, MAM +7.5% -- the two seasons featured on the poster.
"""
import os
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

WIND_PATH = "wind_stress_on_ease_sh.nc"
LATLON_PATH = "ease_divergence_with_latlon.nc"
SPLIT_YEAR = 2016
EXCLUDE_YEARS = [1978, 1987, 1991, 1995]
SEASONS = {"DJF": [12, 1, 2], "MAM": [3, 4, 5]}
QUIVER_SKIP = 20
QUIVER_SCALE = 0.5
QUIVER_COLOR = "0.15"
QUIVER_ALPHA = 0.75

EASE_CRS = ccrs.LambertAzimuthalEqualArea(central_latitude=-90.0, central_longitude=0.0)
PLATE = ccrs.PlateCarree()
OUT = "fig_wind_djf_mam.png"
RCLONE_REMOTE = "gdrive:scar_poster/"


def _tname(da):
    for c in ("time", "valid_time"):
        if c in da.dims:
            return c
    raise KeyError(f"No time dim in {list(da.dims)}")


def seasonal_prepost(da, months):
    tn = _tname(da)
    yrs = da[tn].dt.year
    da = da.sel({tn: ~yrs.isin(EXCLUDE_YEARS)})
    # shift December into the following meteorological-summer year, so
    # DJF season-years group correctly (Dec 2015 + Jan/Feb 2016 = one season)
    adj_year = da[tn].dt.year + (da[tn].dt.month == 12).astype(int)
    da = da.assign_coords(_adj_year=(tn, adj_year.values))
    sub = da.sel({tn: da[tn].dt.month.isin(months)})
    # flat pooled mean over all days (not yearly-mean-of-means) --
    # matches the originally verified aggregation method
    adj_years = sub["_adj_year"].values
    pre = np.nanmean(sub.values[adj_years < SPLIT_YEAR], axis=0)
    post = np.nanmean(sub.values[adj_years >= SPLIT_YEAR], axis=0)
    return pre, post


def main():
    wind_ds = xr.open_dataset(WIND_PATH)
    latlon_ds = xr.open_dataset(LATLON_PATH, decode_times=False)
    x = wind_ds["x"].values
    y = wind_ds["y"].values
    lat2d = latlon_ds["lat"].values
    lon2d = latlon_ds["lon"].values

    cache = {}
    diffs = []
    for season, months in SEASONS.items():
        tx_pre, tx_post = seasonal_prepost(wind_ds["tau_x"], months)
        ty_pre, ty_post = seasonal_prepost(wind_ds["tau_y"], months)
        mag_pre = np.hypot(tx_pre, ty_pre)
        mag_post = np.hypot(tx_post, ty_post)
        diff = mag_post - mag_pre
        band = (lat2d < -50) & (lat2d > -65)
        pct = 100 * (np.nanmean(mag_post[band]) - np.nanmean(mag_pre[band])) / np.nanmean(mag_pre[band])
        print(f"  {season}: {pct:+.2f}%")
        cache[season] = (diff, tx_post - tx_pre, ty_post - ty_pre, pct)
        diffs.append(diff)

    vmax = np.nanpercentile(np.abs(np.concatenate([d.ravel() for d in diffs])), 98)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7),
                             subplot_kw={"projection": EASE_CRS})

    for ax, season in zip(axes, SEASONS.keys()):
        diff, dtx, dty, pct = cache[season]
        im = ax.pcolormesh(x, y, diff, transform=EASE_CRS, cmap="RdBu_r",
                           vmin=-vmax, vmax=vmax, shading="auto", zorder=1)

        s = QUIVER_SKIP
        lon_sub = lon2d[::s, ::s]
        lat_sub = lat2d[::s, ::s]
        u_sub = dtx[::s, ::s].copy()
        v_sub = dty[::s, ::s].copy()
        mag_sub = np.hypot(u_sub, v_sub)
        cap = np.nanpercentile(mag_sub, 95)
        scale_factor = np.where(mag_sub > cap, cap / np.maximum(mag_sub, 1e-12), 1.0)
        u_sub *= scale_factor
        v_sub *= scale_factor
        valid = np.isfinite(lon_sub) & np.isfinite(lat_sub) & np.isfinite(u_sub) & np.isfinite(v_sub)
        lon_sub = np.where(valid, lon_sub, np.nan)
        lat_sub = np.where(valid, lat_sub, np.nan)
        q = ax.quiver(lon_sub, lat_sub, u_sub, v_sub, transform=PLATE,
                     color=QUIVER_COLOR, alpha=QUIVER_ALPHA, scale=QUIVER_SCALE,
                     width=0.0032, headwidth=4, headlength=4, zorder=3)

        ax.add_feature(cfeature.LAND, zorder=4, facecolor="0.85", edgecolor="none")
        ax.coastlines(resolution="50m", linewidth=0.5, zorder=5)
        ax.set_extent([-180, 180, -90, -50], crs=PLATE)
        ax.set_title(f"{season}  ({pct:+.1f}%)", fontsize=16, fontweight="bold")

        ax.quiverkey(q, 0.82, -0.06, cap, f"{cap:.3f} Pa", labelpos="E",
                    coordinates="axes", fontproperties={"size": 10})

    fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.03, pad=0.02,
                label="\u0394 wind stress magnitude (Pa)")
    fig.suptitle("Wind stress strengthened in both seasons", fontsize=17, y=0.99)

    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"-> {OUT}")

    os.system(f"rclone copy {OUT} {RCLONE_REMOTE}")
    print("uploaded.")


if __name__ == "__main__":
    main()
