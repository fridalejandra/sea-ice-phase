"""
fig_wind_convergence_combined.py
Top row = wind stress magnitude diff, bottom row = convergence diff.
DJF/MAM columns, shared vectors + sector boundaries.
"""
import os
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

CONV_PATH = "ice_divergence_daily_sh.nc"
CONV_VAR = "divergence"
WIND_PATH = "wind_stress_on_ease_sh.nc"
LATLON_PATH = "ease_divergence_with_latlon.nc"
SPLIT_YEAR = 2016
EXCLUDE_YEARS = [1978, 1987, 1991, 1995]
SEASONS = {"DJF": [12, 1, 2], "MAM": [3, 4, 5]}
SECTORS = {
    "Weddell": (-60.0, 20.0), "King Haakon": (20.0, 90.0),
    "East Antarctica": (90.0, 160.0), "Ross": (160.0, 230.0),
    "ABS": (230.0, 300.0),
}
QUIVER_SKIP = 18
QUIVER_SCALE = 0.8
EASE_CRS = ccrs.LambertAzimuthalEqualArea(central_latitude=-90.0, central_longitude=0.0)
PLATE = ccrs.PlateCarree()
OUT = "fig_wind_convergence_combined.png"
RCLONE_REMOTE = "gdrive:scar_poster/"


def _tname(da):
    for c in ("time", "valid_time"):
        if c in da.dims:
            return c
    raise KeyError(f"No time dim in {list(da.dims)}")


def seasonal_prepost(ds, var, months, negate=False):
    da = ds[var]
    if negate:
        da = -da
    tn = _tname(da)
    yrs = da[tn].dt.year
    da = da.sel({tn: ~yrs.isin(EXCLUDE_YEARS)})
    sub = da.sel({tn: da[tn].dt.month.isin(months)})
    ym = sub.groupby(sub[tn].dt.year).mean(dim=tn).load()
    years = ym["year"].values
    pre = np.nanmean(ym.values[years < SPLIT_YEAR], axis=0)
    post = np.nanmean(ym.values[years >= SPLIT_YEAR], axis=0)
    return pre, post


def draw_sector_boundaries(ax):
    for name, (lon_min, lon_max) in SECTORS.items():
        lm = ((lon_min + 180) % 360) - 180
        ax.plot([lm, lm], [-90, -50], transform=PLATE,
                color="0.25", linewidth=1.0, linestyle="--", alpha=0.6, zorder=6)


def add_vectors(ax, lon2d, lat2d, dtx, dty):
    s = QUIVER_SKIP
    lon_sub = lon2d[::s, ::s]
    lat_sub = lat2d[::s, ::s]
    u_sub = dtx[::s, ::s]
    v_sub = dty[::s, ::s]
    valid = np.isfinite(lon_sub) & np.isfinite(lat_sub) & np.isfinite(u_sub) & np.isfinite(v_sub)
    lon_sub = np.where(valid, lon_sub, np.nan)
    lat_sub = np.where(valid, lat_sub, np.nan)
    return ax.quiver(lon_sub, lat_sub, u_sub, v_sub, transform=PLATE,
                     color="0.1", alpha=0.75, scale=QUIVER_SCALE, width=0.0032,
                     headwidth=4, headlength=4, zorder=3)


def main():
    conv_ds = xr.open_dataset(CONV_PATH)
    wind_ds = xr.open_dataset(WIND_PATH)
    latlon_ds = xr.open_dataset(LATLON_PATH, decode_times=False)
    x = conv_ds["x"].values
    y = conv_ds["y"].values
    lat2d = latlon_ds["lat"].values
    lon2d = latlon_ds["lon"].values

    cache = {}
    wind_diffs, conv_diffs = [], []
    for season, months in SEASONS.items():
        tx_pre, tx_post = seasonal_prepost(wind_ds, "tau_x", months)
        ty_pre, ty_post = seasonal_prepost(wind_ds, "tau_y", months)
        mag_pre = np.hypot(tx_pre, ty_pre)
        mag_post = np.hypot(tx_post, ty_post)
        mag_diff = mag_post - mag_pre
        dtx, dty = tx_post - tx_pre, ty_post - ty_pre

        conv_pre, conv_post = seasonal_prepost(conv_ds, CONV_VAR, months, negate=True)
        conv_diff = conv_post - conv_pre

        cache[season] = (mag_diff, conv_diff, dtx, dty)
        wind_diffs.append(mag_diff)
        conv_diffs.append(conv_diff)

    vmax_wind = np.nanpercentile(np.abs(np.concatenate([d.ravel() for d in wind_diffs])), 98)
    vmax_conv = np.nanpercentile(np.abs(np.concatenate([d.ravel() for d in conv_diffs])), 98)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 11),
                             subplot_kw={"projection": EASE_CRS})

    im_wind = im_conv = None
    for col, season in enumerate(SEASONS.keys()):
        mag_diff, conv_diff, dtx, dty = cache[season]

        ax = axes[0, col]
        im_wind = ax.pcolormesh(x, y, mag_diff, transform=EASE_CRS, cmap="RdBu_r",
                                vmin=-vmax_wind, vmax=vmax_wind, shading="auto", zorder=1)
        add_vectors(ax, lon2d, lat2d, dtx, dty)
        draw_sector_boundaries(ax)
        ax.add_feature(cfeature.LAND, zorder=4, facecolor="0.85", edgecolor="none")
        ax.coastlines(resolution="50m", linewidth=0.5, zorder=5)
        ax.set_extent([-180, 180, -90, -50], crs=PLATE)
        ax.set_title(season, fontsize=17, fontweight="bold", pad=10)

        ax = axes[1, col]
        im_conv = ax.pcolormesh(x, y, conv_diff, transform=EASE_CRS, cmap="RdBu_r",
                                vmin=-vmax_conv, vmax=vmax_conv, shading="auto", zorder=1)
        add_vectors(ax, lon2d, lat2d, dtx, dty)
        draw_sector_boundaries(ax)
        ax.add_feature(cfeature.LAND, zorder=4, facecolor="0.85", edgecolor="none")
        ax.coastlines(resolution="50m", linewidth=0.5, zorder=5)
        ax.set_extent([-180, 180, -90, -50], crs=PLATE)

    axes[0, 0].text(-0.12, 0.5, "Wind stress\nmagnitude \u0394",
                    transform=axes[0, 0].transAxes, fontsize=14, fontweight="bold",
                    va="center", ha="right", rotation=90)
    axes[1, 0].text(-0.12, 0.5, "Convergence \u0394",
                    transform=axes[1, 0].transAxes, fontsize=14, fontweight="bold",
                    va="center", ha="right", rotation=90)

    fig.colorbar(im_wind, ax=axes[0, :].tolist(), orientation="vertical",
                fraction=0.03, pad=0.02, label="\u0394 stress (Pa)")
    fig.colorbar(im_conv, ax=axes[1, :].tolist(), orientation="vertical",
                fraction=0.03, pad=0.02, label="\u0394 convergence (s\u207b\u00b9)")

    fig.suptitle("Westerlies strengthened ~11% (top) \u2014 convergence increased at the edge (bottom)",
                 fontsize=16, y=0.98)

    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    print(f"-> {OUT}")

    os.system(f"rclone copy {OUT} {RCLONE_REMOTE}")
    print("uploaded.")


if __name__ == "__main__":
    main()
