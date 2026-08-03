"""
fig_sic_wind_overlay.py

Poster section 1 figure: SIC difference (post-pre, color fill) with wind
stress difference overlaid as vectors. One image: ice declined everywhere,
wind pushed harder everywhere, and it didn't matter.
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
SIC_VAR = "sic"

SPLIT_YEAR = 2016
EXCLUDE_YEARS = [1978, 1987, 1991, 1995]
MONTH = 9

# subsample vectors so the plot isn't a wall of arrows
QUIVER_SKIP = 8    # plot every Nth grid cell

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


def monthly_prepost(path, var, month):
    ds = xr.open_dataset(path)
    da = ds[var]
    tn = _tname(da)
    yrs = da[tn].dt.year
    da = da.sel({tn: ~yrs.isin(EXCLUDE_YEARS)})
    sub = da.sel({tn: da[tn].dt.month == month})
    ym = sub.groupby(sub[tn].dt.year).mean(dim=tn).load()
    years = ym["year"].values
    pre = ym.values[years < SPLIT_YEAR]
    post = ym.values[years >= SPLIT_YEAR]
    return np.nanmean(pre, axis=0), np.nanmean(post, axis=0), ds["x"].values, ds["y"].values


def main():
    # SIC difference
    sic_pre, sic_post, x, y = monthly_prepost(SIC_PATH, SIC_VAR, MONTH)
    sic_diff = sic_post - sic_pre

    # wind stress components
    wind_ds = xr.open_dataset(WIND_PATH)
    tn = _tname(wind_ds["tau_x"])
    yrs = wind_ds[tn].dt.year

    wind_ds = wind_ds.sel({tn: ~yrs.isin(EXCLUDE_YEARS)})
    wind_ds = wind_ds.sel({tn: wind_ds[tn].dt.month == MONTH})
    ym_tx = wind_ds["tau_x"].groupby(wind_ds[tn].dt.year).mean(dim=tn).load()
    ym_ty = wind_ds["tau_y"].groupby(wind_ds[tn].dt.year).mean(dim=tn).load()

    years = ym_tx["year"].values
    tx_pre = np.nanmean(ym_tx.values[years < SPLIT_YEAR], axis=0)
    tx_post = np.nanmean(ym_tx.values[years >= SPLIT_YEAR], axis=0)
    ty_pre = np.nanmean(ym_ty.values[years < SPLIT_YEAR], axis=0)
    ty_post = np.nanmean(ym_ty.values[years >= SPLIT_YEAR], axis=0)

    dtx = tx_post - tx_pre
    dty = ty_post - ty_pre
    wx = wind_ds["x"].values
    wy = wind_ds["y"].values

    fig, ax = plt.subplots(1, 1, figsize=(8, 8),
                           subplot_kw={"projection": EASE_CRS})

    # SIC difference as color fill
    vmax_sic = np.nanpercentile(np.abs(sic_diff), 98)
    im = ax.pcolormesh(x, y, sic_diff, transform=EASE_CRS,
                       cmap="RdBu_r", vmin=-vmax_sic, vmax=vmax_sic,
                       shading="auto", zorder=1)

    # wind stress difference as vectors, subsampled
    s = QUIVER_SKIP
    xx, yy = np.meshgrid(wx[::s], wy[::s])
    u = dtx[::s, ::s]
    v = dty[::s, ::s]

    ax.quiver(xx, yy, u, v, transform=EASE_CRS,
              color="black", alpha=0.7, scale=0.03, width=0.003,
              headwidth=4, headlength=5, zorder=3)

    ax.add_feature(cfeature.LAND, zorder=4, facecolor="0.85", edgecolor="none")
    ax.coastlines(resolution="50m", linewidth=0.5, zorder=5)
    ax.set_extent([-180, 180, -90, -50], crs=PLATE)

    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.045,
                      pad=0.05, label="ΔSIC (fraction, post − pre)")

    ax.set_title("September: sea ice declined (color)\n"
                 "wind stress change (vectors)",
                 fontsize=14)

    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"-> {OUT}")

    os.system(f"rclone copy {OUT} {RCLONE_REMOTE}")
    print("uploaded.")


if __name__ == "__main__":
    main()