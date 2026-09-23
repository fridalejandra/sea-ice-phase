"""
fig_wind_sia_khv_stacked.py
MAM only, side by side: wind stress magnitude (left, with vector legend)
+ SIA change (right). Quiver outliers clipped at 95th percentile.
"""
import os
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

SIC_PATH = "sic_bootstrap_on_ease_sh.nc"
WIND_PATH = "wind_stress_on_ease_sh.nc"
LATLON_PATH = "ease_divergence_with_latlon.nc"
SIC_VAR = "sic"
SPLIT_YEAR = 2016
EXCLUDE_YEARS = [1978, 1987, 1991, 1995]
SEASON_MONTHS = [3, 4, 5]
QUIVER_SKIP = 20
QUIVER_SCALE = 0.5
QUIVER_COLOR = "0.15"
QUIVER_ALPHA = 0.75

EASE_CRS = ccrs.LambertAzimuthalEqualArea(central_latitude=-90.0, central_longitude=0.0)
PLATE = ccrs.PlateCarree()
OUT = "fig_wind_sia_khv_stacked.png"
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
    sub = da.sel({tn: da[tn].dt.month.isin(months)})
    ym = sub.groupby(sub[tn].dt.year).mean(dim=tn).load()
    years = ym["year"].values
    pre = np.nanmean(ym.values[years < SPLIT_YEAR], axis=0)
    post = np.nanmean(ym.values[years >= SPLIT_YEAR], axis=0)
    return pre, post


def main():
    sic_ds = xr.open_dataset(SIC_PATH)
    wind_ds = xr.open_dataset(WIND_PATH)
    latlon_ds = xr.open_dataset(LATLON_PATH, decode_times=False)

    x = sic_ds["x"].values
    y = sic_ds["y"].values
    lat2d = latlon_ds["lat"].values
    lon2d = latlon_ds["lon"].values

    tx_pre, tx_post = seasonal_prepost(wind_ds["tau_x"], SEASON_MONTHS)
    ty_pre, ty_post = seasonal_prepost(wind_ds["tau_y"], SEASON_MONTHS)
    mag_pre = np.hypot(tx_pre, ty_pre)
    mag_post = np.hypot(tx_post, ty_post)
    mag_diff = mag_post - mag_pre
    dtx, dty = tx_post - tx_pre, ty_post - ty_pre
    vmax_wind = np.nanpercentile(np.abs(mag_diff), 98)

    sic_pre, sic_post = seasonal_prepost(sic_ds[SIC_VAR], SEASON_MONTHS)
    sia_diff = (sic_post - sic_pre) * 625.0
    vmax_sia = np.nanpercentile(np.abs(sia_diff), 98)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5),
                             subplot_kw={"projection": EASE_CRS})

    ax = axes[0]
    im_wind = ax.pcolormesh(x, y, mag_diff, transform=EASE_CRS, cmap="RdBu_r",
                            vmin=-vmax_wind, vmax=vmax_wind, shading="auto", zorder=1)

    s = QUIVER_SKIP
    lon_sub = lon2d[::s, ::s]
    lat_sub = lat2d[::s, ::s]
    u_sub = dtx[::s, ::s].copy()
    v_sub = dty[::s, ::s].copy()

    mag_sub = np.hypot(u_sub, v_sub)
    cap = np.nanpercentile(mag_sub, 95)
    scale_factor = np.where(mag_sub > cap, cap / np.maximum(mag_sub, 1e-12), 1.0)
    u_sub = u_sub * scale_factor
    v_sub = v_sub * scale_factor

    valid = np.isfinite(lon_sub) & np.isfinite(lat_sub) & np.isfinite(u_sub) & np.isfinite(v_sub)
    lon_sub = np.where(valid, lon_sub, np.nan)
    lat_sub = np.where(valid, lat_sub, np.nan)
    q = ax.quiver(lon_sub, lat_sub, u_sub, v_sub, transform=PLATE,
                 color=QUIVER_COLOR, alpha=QUIVER_ALPHA, scale=QUIVER_SCALE,
                 width=0.0032, headwidth=4, headlength=4, zorder=3)

    ax.add_feature(cfeature.LAND, zorder=4, facecolor="0.85", edgecolor="none")
    ax.coastlines(resolution="50m", linewidth=0.5, zorder=5)
    ax.set_extent([-180, 180, -90, -50], crs=PLATE)
    ax.set_title("Wind stress magnitude \u0394 \u2014 MAM", fontsize=15, fontweight="bold")
    fig.colorbar(im_wind, ax=ax, orientation="vertical", fraction=0.045,
                pad=0.03, label="\u0394 stress (Pa)")

    ax.quiverkey(q, 0.82, -0.06, cap, f"{cap:.3f} Pa", labelpos="E",
                coordinates="axes", fontproperties={"size": 10})

    ax = axes[1]
    im_sia = ax.pcolormesh(x, y, sia_diff, transform=EASE_CRS, cmap="RdBu_r",
                           vmin=-vmax_sia, vmax=vmax_sia, shading="auto", zorder=1)

    ax.add_feature(cfeature.LAND, zorder=4, facecolor="0.85", edgecolor="none")
    ax.coastlines(resolution="50m", linewidth=0.5, zorder=5)
    ax.set_extent([-180, 180, -90, -50], crs=PLATE)
    ax.set_title("Sea ice area \u0394 \u2014 MAM", fontsize=15, fontweight="bold")

    fig.colorbar(im_sia, ax=ax, orientation="vertical", fraction=0.045,
                pad=0.03, label="\u0394SIA per grid cell (km\u00b2)")

    fig.tight_layout()
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"-> {OUT}")

    os.system(f"rclone copy {OUT} {RCLONE_REMOTE}")
    print("uploaded.")


if __name__ == "__main__":
    main()
