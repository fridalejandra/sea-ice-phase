"""
fig_convergence_tendency.py

Poster section 4 figure: two maps side by side, September.
Left: convergence difference (where ridging reorganized at the new edge)
Right: ΔSIC tendency difference (where variability vanished — the red ring)
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

CONV_PATH = "ice_divergence_daily_sh.nc"
CONV_VAR = "divergence"
DSIC_PATH = "dsic_daily_on_ease_sh.nc"
DSIC_VAR = "dsic"

SPLIT_YEAR = 2016
EXCLUDE_YEARS = [1978, 1987, 1991, 1995]
MONTH = 9

EASE_CRS = ccrs.LambertAzimuthalEqualArea(
    central_latitude=-90.0, central_longitude=0.0
)
PLATE = ccrs.PlateCarree()

OUT = "fig_convergence_tendency.png"
RCLONE_REMOTE = "gdrive:scar_poster/"


def _tname(da):
    for c in ("time", "valid_time"):
        if c in da.dims:
            return c
    raise KeyError(f"No time dim in {list(da.dims)}")


def monthly_prepost(path, var, month, negate=False):
    ds = xr.open_dataset(path)
    da = ds[var]
    if negate:
        da = -da
    tn = _tname(da)
    yrs = da[tn].dt.year
    da = da.sel({tn: ~yrs.isin(EXCLUDE_YEARS)})
    sub = da.sel({tn: da[tn].dt.month == month})
    ym = sub.groupby(sub[tn].dt.year).mean(dim=tn).load()
    years = ym["year"].values
    pre = ym.values[years < SPLIT_YEAR]
    post = ym.values[years >= SPLIT_YEAR]
    pre_mean = np.nanmean(pre, axis=0)
    post_mean = np.nanmean(post, axis=0)
    diff = post_mean - pre_mean
    with np.errstate(invalid="ignore"):
        _, pval = stats.ttest_ind(post, pre, axis=0,
                                  equal_var=False, nan_policy="omit")
    pval = np.ma.filled(np.ma.masked_invalid(pval), 1.0)
    return diff, pval, ds["x"].values, ds["y"].values


def main():
    conv_diff, conv_p, x, y = monthly_prepost(CONV_PATH, CONV_VAR, MONTH,
                                               negate=True)
    dsic_diff, dsic_p, dx, dy = monthly_prepost(DSIC_PATH, DSIC_VAR, MONTH)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5),
                             subplot_kw={"projection": EASE_CRS})

    panels = [
        (axes[0], conv_diff, conv_p, x, y,
         "Convergence change", "Δ convergence (s⁻¹)", "RdBu_r"),
        (axes[1], dsic_diff, dsic_p, dx, dy,
         "Tendency (ΔSIC) change", "Δ tendency (fraction/day)", "RdBu_r"),
    ]

    for ax, diff, pval, px, py, title, clabel, cmap in panels:
        vmax = np.nanpercentile(np.abs(diff), 98)
        if not np.isfinite(vmax) or vmax == 0:
            vmax = 1e-9
        im = ax.pcolormesh(px, py, diff, transform=EASE_CRS,
                           cmap=cmap, vmin=-vmax, vmax=vmax,
                           shading="auto", zorder=1)

        sig = pval < 0.05
        yy, xx = np.meshgrid(py, px, indexing="ij")
        ax.scatter(xx[sig], yy[sig], s=0.12, c="k", alpha=0.45,
                   transform=EASE_CRS, zorder=5, linewidths=0)

        ax.add_feature(cfeature.LAND, zorder=4, facecolor="0.85",
                       edgecolor="none")
        ax.coastlines(resolution="50m", linewidth=0.5, zorder=5)
        ax.set_extent([-180, 180, -90, -50], crs=PLATE)
        ax.set_title(title, fontsize=14, fontweight="bold")

        fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.045,
                     pad=0.05, label=clabel, shrink=0.9)

    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"-> {OUT}")

    os.system(f"rclone copy {OUT} {RCLONE_REMOTE}")
    print("uploaded.")


if __name__ == "__main__":
    main()