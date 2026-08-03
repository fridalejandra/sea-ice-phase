"""
fig_sic_wind_overlay.py

Poster section 1: SIC difference (color) + wind stress difference (vectors),
two panels — March (minimum extent) and September (maximum extent).
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
MONTHS = [3, 9]
MONTH_NAMES = {3: "March (min extent)", 9: "September (max extent)"}

QUIVER_SKIP = 12
QUIVER_SCALE = 0.04
QUIVER_WIDTH = 0.003
QUIVER_COLOR = "0.15"
QUIVER_ALPHA = 0.6

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


def monthly_prepost_field(ds, var, month):
    da = ds[var]
    tn = _tname(da)
    yrs = da[tn].dt.year
    da = da.sel({tn: ~yrs.isin(EXCLUDE_YEARS)})
    sub = da.sel({tn: da[tn].dt.month == month})
    ym = sub.groupby(sub[tn].dt.year).mean(dim=tn).load()
    years = ym["year"].values
    pre = np.nanmean(ym.values[years < SPLIT_YEAR], axis=0)
    post = np.nanmean(ym.values[years >= SPLIT_YEAR], axis=0)
    return pre, post


def main():
    sic_ds = xr.open_dataset(SIC_PATH)
    wind_ds = xr.open_dataset(WIND_PATH)
    x_sic = sic_ds["x"].values
    y_sic = sic_ds["y"].values
    x_wind = wind_ds["x"].values
    y_wind = wind_ds["y"].values

    # compute global vmax across both months for consistent color scale
    all_diffs = []
    for month in MONTHS:
        pre, post = monthly_prepost_field(sic_ds, SIC_VAR, month)
        all_diffs.append(post - pre)
    vmax_sic = np.nanpercentile(np.abs(np.concatenate([d.ravel() for d in all_diffs])), 98)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7),
                             subplot_kw={"projection": EASE_CRS})

    for ax, month in zip(axes, MONTHS):
        # SIC difference
        sic_pre, sic_post = monthly_prepost_field(sic_ds, SIC_VAR, month)
        sic_diff = sic_post - sic_pre

        im = ax.pcolormesh(x_sic, y_sic, sic_diff, transform=EASE_CRS,
                           cmap="RdBu_r", vmin=-vmax_sic, vmax=vmax_sic,
                           shading="auto", zorder=1)

        # wind stress vectors
        tx_pre, tx_post = monthly_prepost_field(wind_ds, "tau_x", month)
        ty_pre, ty_post = monthly_prepost_field(wind_ds, "tau_y", month)
        dtx = tx_post - tx_pre
        dty = ty_post - ty_pre

        s = QUIVER_SKIP
        xx, yy = np.meshgrid(x_wind[::s], y_wind[::s])
        u = dtx[::s, ::s]
        v = dty[::s, ::s]

        q = ax.quiver(xx, yy, u, v, transform=EASE_CRS,
                      color=QUIVER_COLOR, alpha=QUIVER_ALPHA,
                      scale=QUIVER_SCALE, width=QUIVER_WIDTH,
                      headwidth=4, headlength=4, zorder=3)

        ax.add_feature(cfeature.LAND, zorder=4, facecolor="0.85",
                       edgecolor="none")
        ax.coastlines(resolution="50m", linewidth=0.5, zorder=5)
        ax.set_extent([-180, 180, -90, -50], crs=PLATE)
        ax.set_title(MONTH_NAMES[month], fontsize=13)

    # reference arrow
    axes[1].quiverkey(q, 0.85, 0.02, 0.002, "0.002 Pa",
                      labelpos="E", fontproperties={"size": 10})

    # shared colorbar
    fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.04,
                 pad=0.06, label="ΔSIC (fraction, post − pre)")

    fig.suptitle("Sea ice declined (color) while wind stress changed (vectors)",
                 fontsize=15, y=0.98)

    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"-> {OUT}")

    os.system(f"rclone copy {OUT} {RCLONE_REMOTE}")
    print("uploaded.")


if __name__ == "__main__":
    main()