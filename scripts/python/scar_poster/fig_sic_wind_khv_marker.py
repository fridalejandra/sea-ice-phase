"""
fig_sic_wind_khv_marker.py
SIC + wind overlay with KHV marker on MAM panel only.
"""
import os
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import cartopy.crs as ccrs
import cartopy.feature as cfeature

SIC_PATH = "sic_bootstrap_on_ease_sh.nc"
WIND_PATH = "wind_stress_on_ease_sh.nc"
LATLON_PATH = "ease_divergence_with_latlon.nc"
SIC_VAR = "sic"
SPLIT_YEAR = 2016
EXCLUDE_YEARS = [1978, 1987, 1991, 1995]
SEASONS = {"DJF": [12, 1, 2], "MAM": [3, 4, 5]}
QUIVER_SKIP = 12
QUIVER_SCALE = 0.025
QUIVER_COLOR = "0.2"
QUIVER_ALPHA = 0.65
KHV_MARKER_LON = 55.0
KHV_MARKER_LAT = -63.0

EASE_CRS = ccrs.LambertAzimuthalEqualArea(central_latitude=-90.0, central_longitude=0.0)
PLATE = ccrs.PlateCarree()
OUT = "fig_sic_wind_khv_marker.png"
RCLONE_REMOTE = "gdrive:scar_poster/"


def _tname(da):
    for c in ("time", "valid_time"):
        if c in da.dims:
            return c
    raise KeyError(f"No time dim in {list(da.dims)}")


def seasonal_prepost_field(ds, var, months):
    da = ds[var]
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

    x_sic = sic_ds["x"].values
    y_sic = sic_ds["y"].values
    lat2d = latlon_ds["lat"].values
    lon2d = latlon_ds["lon"].values

    all_diffs = []
    for season, months in SEASONS.items():
        pre, post = seasonal_prepost_field(sic_ds, SIC_VAR, months)
        all_diffs.append((post - pre) * 625.0)
    vmax_sic = np.nanpercentile(np.abs(np.concatenate(
        [d.ravel() for d in all_diffs])), 98)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7),
                             subplot_kw={"projection": EASE_CRS})

    for ax, (season, months) in zip(axes, SEASONS.items()):
        sic_pre, sic_post = seasonal_prepost_field(sic_ds, SIC_VAR, months)
        sic_diff = (sic_post - sic_pre) * 625.0

        im = ax.pcolormesh(x_sic, y_sic, sic_diff, transform=EASE_CRS,
                           cmap="RdBu_r", vmin=-vmax_sic, vmax=vmax_sic,
                           shading="auto", zorder=1)

        tx_pre, tx_post = seasonal_prepost_field(wind_ds, "tau_x", months)
        ty_pre, ty_post = seasonal_prepost_field(wind_ds, "tau_y", months)
        dtx = tx_post - tx_pre
        dty = ty_post - ty_pre

        s = QUIVER_SKIP
        lon_sub = lon2d[::s, ::s]
        lat_sub = lat2d[::s, ::s]
        u_sub = dtx[::s, ::s]
        v_sub = dty[::s, ::s]
        valid = np.isfinite(lon_sub) & np.isfinite(lat_sub) & np.isfinite(u_sub) & np.isfinite(v_sub)
        lon_sub = np.where(valid, lon_sub, np.nan)
        lat_sub = np.where(valid, lat_sub, np.nan)

        q = ax.quiver(lon_sub, lat_sub, u_sub, v_sub, transform=PLATE,
                      color=QUIVER_COLOR, alpha=QUIVER_ALPHA,
                      scale=QUIVER_SCALE, width=0.003,
                      headwidth=4, headlength=4, regrid_shape=25, zorder=3)

        ax.add_feature(cfeature.LAND, zorder=4, facecolor="0.85", edgecolor="none")
        ax.coastlines(resolution="50m", linewidth=0.5, zorder=5)
        ax.set_extent([-180, 180, -90, -50], crs=PLATE)
        ax.set_title(season, fontsize=15, fontweight="bold")

        if season == "MAM":
            ax.scatter([KHV_MARKER_LON], [KHV_MARKER_LAT], s=500,
                      facecolor="none", edgecolor="black", linewidth=2.5,
                      transform=PLATE, zorder=10)
            ax.text(KHV_MARKER_LON, KHV_MARKER_LAT, "*",
                   transform=PLATE, fontsize=24, ha="center", va="center",
                   fontweight="bold", zorder=11,
                   path_effects=[pe.withStroke(linewidth=3, foreground="white")])
            ax.text(KHV_MARKER_LON, KHV_MARKER_LAT - 8, "KH convergence\ncoupling shifted",
                   transform=PLATE, fontsize=10, ha="center", va="top",
                   fontweight="bold", zorder=11,
                   path_effects=[pe.withStroke(linewidth=3, foreground="white")])

    axes[1].quiverkey(q, 0.85, 0.02, 0.02, "0.02 Pa", labelpos="E",
                      fontproperties={"size": 10})

    fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.025,
                pad=0.02, label="\u0394SIA per grid cell (km\u00b2, post \u2212 pre)")

    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"-> {OUT}")

    os.system(f"rclone copy {OUT} {RCLONE_REMOTE}")
    print("uploaded.")


if __name__ == "__main__":
    main()
