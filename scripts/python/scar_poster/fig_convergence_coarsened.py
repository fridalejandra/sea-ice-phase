"""
fig_convergence_coarsened.py
Real spatial map via coarsened grid to stabilize the mean.
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
SPLIT_YEAR = 2016
EXCLUDE_YEARS = [1978, 1987, 1991, 1995]
SEASONS = {"DJF": [12, 1, 2], "MAM": [3, 4, 5]}
COARSEN_FACTOR = 5
MIN_VALID_FRAC = 0.15

EASE_CRS = ccrs.LambertAzimuthalEqualArea(central_latitude=-90.0, central_longitude=0.0)
PLATE = ccrs.PlateCarree()
OUT = "fig_convergence_coarsened.png"
RCLONE_REMOTE = "gdrive:scar_poster/"


def _tname(da):
    for c in ("time", "valid_time"):
        if c in da.dims:
            return c
    raise KeyError(f"No time dim in {list(da.dims)}")


def coarsen_spatial(da, factor):
    ny = (da.sizes["y"] // factor) * factor
    nx = (da.sizes["x"] // factor) * factor
    da = da.isel(y=slice(0, ny), x=slice(0, nx))
    return da.coarsen(y=factor, x=factor, boundary="trim")


def seasonal_prepost_coarse(ds, var, months, negate=False):
    da = ds[var]
    if negate:
        da = -da
    tn = _tname(da)
    yrs = da[tn].dt.year
    da = da.sel({tn: ~yrs.isin(EXCLUDE_YEARS)})
    sub = da.sel({tn: da[tn].dt.month.isin(months)})

    post_mask = sub[tn].dt.year.values >= SPLIT_YEAR
    pre_da = sub.isel({tn: ~post_mask})
    post_da = sub.isel({tn: post_mask})

    pre_coarse = coarsen_spatial(pre_da, COARSEN_FACTOR).mean()
    post_coarse = coarsen_spatial(post_da, COARSEN_FACTOR).mean()

    pre_valid = coarsen_spatial(np.isfinite(pre_da).astype(float), COARSEN_FACTOR).mean()
    post_valid = coarsen_spatial(np.isfinite(post_da).astype(float), COARSEN_FACTOR).mean()

    pre_mean = pre_coarse.mean(dim=tn).values
    post_mean = post_coarse.mean(dim=tn).values
    pre_vfrac = pre_valid.mean(dim=tn).values
    post_vfrac = post_valid.mean(dim=tn).values

    keep = (pre_vfrac >= MIN_VALID_FRAC) & (post_vfrac >= MIN_VALID_FRAC)
    pre_mean = np.where(keep, pre_mean, np.nan)
    post_mean = np.where(keep, post_mean, np.nan)

    print(f"    kept {keep.sum()}/{keep.size} coarsened cells "
          f"({100*keep.sum()/keep.size:.1f}%)")

    nx = (ds.sizes["x"] // COARSEN_FACTOR) * COARSEN_FACTOR
    ny = (ds.sizes["y"] // COARSEN_FACTOR) * COARSEN_FACTOR
    x_coarse = ds["x"].isel(x=slice(0, nx)).coarsen(x=COARSEN_FACTOR, boundary="trim").mean().values
    y_coarse = ds["y"].isel(y=slice(0, ny)).coarsen(y=COARSEN_FACTOR, boundary="trim").mean().values

    return pre_mean, post_mean, x_coarse, y_coarse


def main():
    conv_ds = xr.open_dataset(CONV_PATH)

    cache = {}
    diffs = []
    for season, months in SEASONS.items():
        print(f"{season}:")
        pre, post, xc, yc = seasonal_prepost_coarse(conv_ds, CONV_VAR, months, negate=True)
        diff = post - pre
        cache[season] = (diff, xc, yc)
        diffs.append(diff)

    vmax = np.nanpercentile(np.abs(np.concatenate([d.ravel() for d in diffs])), 95)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5),
                             subplot_kw={"projection": EASE_CRS})

    for ax, season in zip(axes, SEASONS.keys()):
        diff, xc, yc = cache[season]
        im = ax.pcolormesh(xc, yc, diff, transform=EASE_CRS, cmap="RdBu_r",
                           vmin=-vmax, vmax=vmax, shading="auto", zorder=1)
        ax.add_feature(cfeature.LAND, zorder=4, facecolor="0.85", edgecolor="none")
        ax.coastlines(resolution="50m", linewidth=0.5, zorder=5)
        ax.set_extent([-180, 180, -90, -50], crs=PLATE)
        ax.set_title(season, fontsize=16, fontweight="bold")

    fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.03, pad=0.02,
                label="\u0394 convergence (s\u207b\u00b9)")
    fig.suptitle(f"Convergence change, {COARSEN_FACTOR*25}km blocks, "
                 f"\u2265{int(MIN_VALID_FRAC*100)}% ice-day coverage required",
                 fontsize=14, y=1.02)

    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    print(f"-> {OUT}")

    os.system(f"rclone copy {OUT} {RCLONE_REMOTE}")
    print("uploaded.")


if __name__ == "__main__":
    main()
