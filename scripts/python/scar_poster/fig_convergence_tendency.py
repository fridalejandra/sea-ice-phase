"""
fig_convergence_seasonal.py

Convergence difference (post - pre 2016) by season (DJF, MAM, JJA, SON).
One row, four panels. See if the signal is cleaner seasonally than monthly.
"""

import os
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy import stats

PATH = "ice_divergence_daily_sh.nc"
VAR = "divergence"

SPLIT_YEAR = 2016
EXCLUDE_YEARS = [1978, 1987, 1991, 1995]

SEASONS = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
}

EASE_CRS = ccrs.LambertAzimuthalEqualArea(
    central_latitude=-90.0, central_longitude=0.0
)
PLATE = ccrs.PlateCarree()

OUT = "fig_convergence_seasonal.png"
RCLONE_REMOTE = "gdrive:scar_poster/"


def main():
    ds = xr.open_dataset(PATH)
    da = -ds[VAR]  # negate for convergence
    yrs = da["time"].dt.year
    da = da.sel(time=~yrs.isin(EXCLUDE_YEARS))
    x = ds["x"].values
    y = ds["y"].values

    fig, axes = plt.subplots(1, 4, figsize=(18, 5),
                             subplot_kw={"projection": EASE_CRS})

    # compute global vmax across all seasons
    all_diffs = []
    for season, months in SEASONS.items():
        sub = da.sel(time=da["time"].dt.month.isin(months))
        ym = sub.groupby(sub["time"].dt.year).mean(dim="time").load()
        years = ym["year"].values
        pre = np.nanmean(ym.values[years < SPLIT_YEAR], axis=0)
        post = np.nanmean(ym.values[years >= SPLIT_YEAR], axis=0)
        all_diffs.append(post - pre)
    vmax = np.nanpercentile(np.abs(np.concatenate([d.ravel() for d in all_diffs])), 98)

    for ax, (season, months) in zip(axes, SEASONS.items()):
        sub = da.sel(time=da["time"].dt.month.isin(months))
        ym = sub.groupby(sub["time"].dt.year).mean(dim="time").load()
        years = ym["year"].values
        pre = ym.values[years < SPLIT_YEAR]
        post = ym.values[years >= SPLIT_YEAR]
        diff = np.nanmean(post, axis=0) - np.nanmean(pre, axis=0)

        with np.errstate(invalid="ignore"):
            _, pval = stats.ttest_ind(post, pre, axis=0,
                                      equal_var=False, nan_policy="omit")
        pval = np.ma.filled(np.ma.masked_invalid(pval), 1.0)

        im = ax.pcolormesh(x, y, diff, transform=EASE_CRS,
                           cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                           shading="auto", zorder=1)

        sig = pval < 0.05
        yy, xx = np.meshgrid(y, x, indexing="ij")
        ax.scatter(xx[sig], yy[sig], s=0.1, c="k", alpha=0.4,
                   transform=EASE_CRS, zorder=5, linewidths=0)

        ax.add_feature(cfeature.LAND, zorder=4, facecolor="0.85",
                       edgecolor="none")
        ax.coastlines(resolution="50m", linewidth=0.5, zorder=5)
        ax.set_extent([-180, 180, -90, -50], crs=PLATE)
        ax.set_title(season, fontsize=14, fontweight="bold")

    fig.colorbar(im, ax=axes.ravel().tolist(), orientation="horizontal",
                 fraction=0.04, pad=0.06,
                 label="Δ convergence (s⁻¹, post − pre)")

    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"-> {OUT}")

    os.system(f"rclone copy {OUT} {RCLONE_REMOTE}")
    print("uploaded.")


if __name__ == "__main__":
    main()