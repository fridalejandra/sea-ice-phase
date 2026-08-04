"""
fig_sic_wind_overlay.py

Poster section 1: SIC difference (color) + wind stress difference (vectors),
two panels — DJF and MAM composites. These are the seasons where the wind
stress trend is actually significant (Mann-Kendall, FDR-corrected), so the
figure matches the stated result rather than an arbitrary month choice.
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

SEASONS = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
}

QUIVER_SKIP = 12
QUIVER_SCALE = 0.008
QUIVER_WIDTH = 0.0025
QUIVER_COLOR = "black"
QUIVER_ALPHA = 0.8

EASE_CRS = ccrs.LambertAzimuthalEqualArea(
    central_latitude=-90.0, central_longitude=0.0
)
PLATE = ccrs.PlateCarree()

OUT = "fig_sic_wind_overlay.png"
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

    # global vmax across both seasons
    all_diffs = []
    for season, months in SEASONS.items():
        pre, post = seasonal_prepost_field(sic_ds, SIC_VAR, months)
        all_diffs.append(post - pre)
    vmax_sic = np.nanpercentile(np.abs(np.concatenate(
        [d.ravel() for d in all_diffs])), 98)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7),
                             subplot_kw={"projection": EASE_CRS})

    for ax, (season, months) in zip(axes, SEASONS.items()):
        # SIC difference
        sic_pre, sic_post = seasonal_prepost_field(sic_ds, SIC_VAR, months)
        sic_diff = sic_post - sic_pre

        im = ax.pcolormesh(x_sic, y_sic, sic_diff, transform=EASE_CRS,
                           cmap="RdBu_r", vmin=-vmax_sic, vmax=vmax_sic,
                           shading="auto", zorder=1)

        # wind stress vectors
        tx_pre, tx_post = seasonal_prepost_field(wind_ds, "tau_x", months)
        ty_pre, ty_post = seasonal_prepost_field(wind_ds, "tau_y", months)
        dtx = tx_post - tx_pre
        dty = ty_post - ty_pre

        s = QUIVER_SKIP
        lon_sub = lon2d[::s, ::s]
        lat_sub = lat2d[::s, ::s]
        u_sub = dtx[::s, ::s]
        v_sub = dty[::s, ::s]

        valid = np.isfinite(lon_sub) & np.isfinite(lat_sub) & \
                np.isfinite(u_sub) & np.isfinite(v_sub)
        lon_sub = np.where(valid, lon_sub, np.nan)
        lat_sub = np.where(valid, lat_sub, np.nan)

        q = ax.quiver(lon_sub, lat_sub, u_sub, v_sub,
                      transform=PLATE,
                      color=QUIVER_COLOR, alpha=QUIVER_ALPHA,
                      scale=QUIVER_SCALE, width=QUIVER_WIDTH,
                      headwidth=4, headlength=4,
                      regrid_shape=25,
                      zorder=3)

        ax.add_feature(cfeature.LAND, zorder=4, facecolor="0.85",
                       edgecolor="none")
        ax.coastlines(resolution="50m", linewidth=0.5, zorder=5)
        ax.set_extent([-180, 180, -90, -50], crs=PLATE)
        ax.set_title(season, fontsize=15, fontweight="bold")

    axes[1].quiverkey(q, 0.85, 0.02, 0.002, "0.002 Pa",
                      labelpos="E", fontproperties={"size": 10})

    fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.04,
                 pad=0.06, label="ΔSIC (fraction, post − pre)")

    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"-> {OUT}")

    os.system(f"rclone copy {OUT} {RCLONE_REMOTE}")
    print("uploaded.")


if __name__ == "__main__":
    main()