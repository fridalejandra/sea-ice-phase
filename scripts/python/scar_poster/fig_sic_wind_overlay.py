"""
fig_sic_wind_overlay.py

Poster section 1 figure: SIC difference (post-pre, color fill) with wind
stress magnitude difference overlaid as contours. One image, one message:
the ice declined everywhere, wind pushed harder everywhere, and it didn't
matter.

September (max extent, biggest contrast).
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
WIND_VAR = "tau_mag"

SPLIT_YEAR = 2016
EXCLUDE_YEARS = [1978, 1987, 1991, 1995]
MONTH = 9

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
    sic_pre, sic_post, x, y = monthly_prepost(SIC_PATH, SIC_VAR, MONTH)
    wind_pre, wind_post, wx, wy = monthly_prepost(WIND_PATH, WIND_VAR, MONTH)

    sic_diff = sic_post - sic_pre
    wind_diff = wind_post - wind_pre

    fig, ax = plt.subplots(1, 1, figsize=(8, 8),
                           subplot_kw={"projection": EASE_CRS})

    # SIC difference as color fill
    vmax_sic = np.nanpercentile(np.abs(sic_diff), 98)
    im = ax.pcolormesh(x, y, sic_diff, transform=EASE_CRS,
                       cmap="RdBu_r", vmin=-vmax_sic, vmax=vmax_sic,
                       shading="auto", zorder=1)

    # wind stress difference as contours overlaid
    # positive contours = wind stress increased
    levels = np.linspace(0.0005, 0.003, 6)
    cs = ax.contour(wx, wy, wind_diff, levels=levels,
                    colors="black", linewidths=0.8, linestyles="-",
                    transform=EASE_CRS, zorder=3)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.4f")

    # negative contours = wind stress decreased (dashed)
    levels_neg = np.linspace(-0.003, -0.0005, 6)
    cs_neg = ax.contour(wx, wy, wind_diff, levels=levels_neg,
                        colors="black", linewidths=0.6, linestyles="--",
                        transform=EASE_CRS, zorder=3)

    ax.add_feature(cfeature.LAND, zorder=4, facecolor="0.85", edgecolor="none")
    ax.coastlines(resolution="50m", linewidth=0.5, zorder=5)
    ax.set_extent([-180, 180, -90, -50], crs=PLATE)

    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.045,
                      pad=0.05, label="ΔSIC (fraction, post − pre)")

    ax.set_title("September: sea ice declined (color)\n"
                 "wind stress increased (contours)",
                 fontsize=14)

    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"-> {OUT}")

    os.system(f"rclone copy {OUT} {RCLONE_REMOTE}")
    print("uploaded.")


if __name__ == "__main__":
    main()