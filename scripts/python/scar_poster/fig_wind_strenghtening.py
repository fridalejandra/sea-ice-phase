"""
fig_div_conv_2x2.py

OPTION B: 2x2 -- rows = DJF/MAM, columns = divergence/convergence.
Difference-only (post-pre), wind vectors + sector boundaries on every
panel. The "even though coupling held, the mean fields still shifted"
figure -- pairs with fig_wind_strengthening.py and the results grid.
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
QUANTITIES = [("divergence", False), ("convergence", True)]

SECTORS = {
    "Weddell":         (-60.0,  20.0),
    "King Haakon":     ( 20.0,  90.0),
    "East Antarctica": ( 90.0, 160.0),
    "Ross":            (160.0, 230.0),
    "ABS":             (230.0, 300.0),
}

QUIVER_SKIP = 18
QUIVER_SCALE = 0.8
QUIVER_COLOR = "0.1"
QUIVER_ALPHA = 0.7

EASE_CRS = ccrs.LambertAzimuthalEqualArea(central_latitude=-90.0, central_longitude=0.0)
PLATE = ccrs.PlateCarree()

OUT = "fig_div_conv_2x2.png"
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
                color="0.25", linewidth=1.1, linestyle="--", alpha=0.7, zorder=6)


def main():
    conv_ds = xr.open_dataset(CONV_PATH)
    wind_ds = xr.open_dataset(WIND_PATH)
    latlon_ds = xr.open_dataset(LATLON_PATH, decode_times=False)
    x = conv_ds["x"].values
    y = conv_ds["y"].values
    lat2d = latlon_ds["lat"].values
    lon2d = latlon_ds["lon"].values

    # separate vmax per quantity (divergence and convergence can differ in scale)
    vmax_by_q = {}
    for qname, negate in QUANTITIES:
        diffs = []
        for season, months in SEASONS.items():
            pre, post = seasonal_prepost(conv_ds, CONV_VAR, months, negate=negate)
            diffs.append(post - pre)
        vmax_by_q[qname] = np.nanpercentile(np.abs(np.concatenate([d.ravel() for d in diffs])), 98)

    fig, axes = plt.subplots(2, 2, figsize=(13, 13),
                             subplot_kw={"projection": EASE_CRS})

    for row, (season, months) in enumerate(SEASONS.items()):
        tx_pre, tx_post = seasonal_prepost(wind_ds, "tau_x", months)
        ty_pre, ty_post = seasonal_prepost(wind_ds, "tau_y", months)
        dtx, dty = tx_post - tx_pre, ty_post - ty_pre

        for col, (qname, negate) in enumerate(QUANTITIES):
            ax = axes[row, col]
            pre, post = seasonal_prepost(conv_ds, CONV_VAR, months, negate=negate)
            diff = post - pre
            vmax = vmax_by_q[qname]

            im = ax.pcolormesh(x, y, diff, transform=EASE_CRS, cmap="RdBu_r",
                               vmin=-vmax, vmax=vmax, shading="auto", zorder=1)

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
                         headwidth=4, headlength=4, zorder=3)

            draw_sector_boundaries(ax)
            ax.add_feature(cfeature.LAND, zorder=4, facecolor="0.85", edgecolor="none")
            ax.coastlines(resolution="50m", linewidth=0.5, zorder=5)
            ax.set_extent([-180, 180, -90, -50], crs=PLATE)

            if row == 0:
                ax.set_title(qname.capitalize(), fontsize=16, fontweight="bold")
            if col == 0:
                ax.text(-0.08, 0.5, season, transform=ax.transAxes, fontsize=15,
                       fontweight="bold", va="center", ha="right", rotation=90)

            fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.045,
                        pad=0.03, label=f"\u0394 {qname} (s\u207b\u00b9)")

    fig.suptitle("Mean fields shifted even though the wind\u2013ice coupling held",
                 fontsize=17, y=0.995)
    fig.tight_layout()
    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    print(f"-> {OUT}")

    os.system(f"rclone copy {OUT} {RCLONE_REMOTE}")
    print("uploaded.")


if __name__ == "__main__":
    main()