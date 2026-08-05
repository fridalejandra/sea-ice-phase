"""
fig_convergence_wind_overlay_seasonal.py

Poster figure: convergence (color) + wind stress vectors, ALL FOUR SEASONS,
pre-mean | post-mean | difference, with sector boundaries overlaid.

Replaces the SIC-based overlay so this figure tests the SAME quantity as
the results grid (divergence-family, sign-flipped for convergence).
Wind vectors on the difference panel visualize the ~11% westerly increase.
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

# set which deformation quantity to plot: "convergence" (negate=True)
# or "divergence" (negate=False) -- run this script twice, once per value
QUANTITY = "convergence"   # "convergence" or "divergence"
NEGATE = (QUANTITY == "convergence")
WIND_PATH = "wind_stress_on_ease_sh.nc"
LATLON_PATH = "ease_divergence_with_latlon.nc"

SPLIT_YEAR = 2016
EXCLUDE_YEARS = [1978, 1987, 1991, 1995]
SEASONS = {"DJF": [12, 1, 2], "MAM": [3, 4, 5],
           "JJA": [6, 7, 8], "SON": [9, 10, 11]}

# same sector boundaries as fig_sector_map_poster.py
SECTORS = {
    "Weddell":         (-60.0,  20.0),
    "King Haakon":     ( 20.0,  90.0),
    "East Antarctica": ( 90.0, 160.0),
    "Ross":            (160.0, 230.0),
    "ABS":             (230.0, 300.0),
}

QUIVER_SKIP = 14
QUIVER_SCALE = 0.8
QUIVER_COLOR = "0.15"
QUIVER_ALPHA = 0.55

EASE_CRS = ccrs.LambertAzimuthalEqualArea(central_latitude=-90.0, central_longitude=0.0)
PLATE = ccrs.PlateCarree()

OUT = f"fig_{QUANTITY}_wind_overlay_seasonal.png"
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
                color="black", linewidth=0.7, alpha=0.8, zorder=6)


def main():
    conv_ds = xr.open_dataset(CONV_PATH)
    wind_ds = xr.open_dataset(WIND_PATH)
    latlon_ds = xr.open_dataset(LATLON_PATH, decode_times=False)
    x = conv_ds["x"].values
    y = conv_ds["y"].values
    lat2d = latlon_ds["lat"].values
    lon2d = latlon_ds["lon"].values

    # global vmax across all seasons' differences
    all_diffs = []
    for season, months in SEASONS.items():
        pre, post = seasonal_prepost(conv_ds, CONV_VAR, months, negate=NEGATE)
        all_diffs.append(post - pre)
    vmax = np.nanpercentile(np.abs(np.concatenate([d.ravel() for d in all_diffs])), 98)
    # shared scale for the mean panels too
    all_means = []
    for season, months in SEASONS.items():
        pre, post = seasonal_prepost(conv_ds, CONV_VAR, months, negate=NEGATE)
        all_means.extend([pre, post])
    vmax_mean = np.nanpercentile(np.abs(np.concatenate([d.ravel() for d in all_means])), 98)

    fig, axes = plt.subplots(4, 3, figsize=(15, 18),
                             subplot_kw={"projection": EASE_CRS})

    for row, (season, months) in enumerate(SEASONS.items()):
        pre, post = seasonal_prepost(conv_ds, CONV_VAR, months, negate=NEGATE)
        diff = post - pre

        tx_pre, tx_post = seasonal_prepost(wind_ds, "tau_x", months)
        ty_pre, ty_post = seasonal_prepost(wind_ds, "tau_y", months)
        dtx, dty = tx_post - tx_pre, ty_post - ty_pre

        panels = [(pre, f"{season}\npre-2016 mean", vmax_mean),
                  (post, "post-2016 mean", vmax_mean),
                  (diff, "difference (post - pre)", vmax)]

        for col, (field, title, vm) in enumerate(panels):
            ax = axes[row, col]
            im = ax.pcolormesh(x, y, field, transform=EASE_CRS, cmap="RdBu_r",
                               vmin=-vm, vmax=vm, shading="auto", zorder=1)

            if col == 2:  # wind vectors only on the difference panel
                s = QUIVER_SKIP
                lon_sub = lon2d[::s, ::s]
                lat_sub = lat2d[::s, ::s]
                u_sub = dtx[::s, ::s]
                v_sub = dty[::s, ::s]
                q = ax.quiver(lon_sub, lat_sub, u_sub, v_sub, transform=PLATE,
                             color=QUIVER_COLOR, alpha=QUIVER_ALPHA,
                             scale=QUIVER_SCALE, width=0.0028,
                             headwidth=4, headlength=4, regrid_shape=22, zorder=3)

            draw_sector_boundaries(ax)
            ax.add_feature(cfeature.LAND, zorder=4, facecolor="0.85", edgecolor="none")
            ax.coastlines(resolution="50m", linewidth=0.4, zorder=5)
            ax.set_extent([-180, 180, -90, -50], crs=PLATE)
            ax.set_title(title, fontsize=11)

        fig.colorbar(im, ax=axes[row, 2], orientation="vertical",
                     fraction=0.045, pad=0.03, label=f"{QUANTITY} (s⁻¹)")

    fig.suptitle(f"{QUANTITY.capitalize()}: pre / post / difference, all seasons\n"
                 "wind-stress change (vectors) on difference panels",
                 fontsize=15, y=0.995)
    fig.tight_layout()
    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    print(f"-> {OUT}")

    os.system(f"rclone copy {OUT} {RCLONE_REMOTE}")
    print("uploaded.")


if __name__ == "__main__":
    main()