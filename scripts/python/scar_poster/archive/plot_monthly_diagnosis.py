"""
plot_monthly_spatial_diagnostics.py

Monthly-resolution spatial diagnostic maps for the Ch4 variables (sea ice
divergence, wind stress [TODO: source], SST anomaly), computed BEFORE
sector/season aggregation.

WHY THIS EXISTS
wind_divergence_coupling_test.py collapses everything to 5 sectors x 4
seasons (20 cells) before testing anything. That collapse assumes each
sector is spatially homogeneous enough for a single wind-divergence
relationship to be meaningful, and assumes DJF/MAM/JJA/SON is the right
temporal grain. Neither assumption has actually been checked against the
gridded fields. This script checks both, at monthly resolution, by mapping:

  1. Pre-2016 vs post-2016 mean fields, per calendar month
  2. The difference (post - pre), with a per-cell screening test
     (NOT a replacement for the sector-level block-bootstrap tests --
     daily-cell autocorrelation makes a naive per-cell t-test overconfident;
     this is a spatial screen to see WHERE a signal might live, not a
     substitute significance test)
  3. (SST only) valid-cell fraction per month, to see the winter
     sea-ice-masking problem directly rather than inferring it from a
     coverage-threshold table

STATUS: diagnostic-grade, not a publication figure. Fast to generate,
meant to sanity-check whether the sector x season collapse is hiding
spatial or seasonal structure before trusting the CSV-based tests.

TODO before running:
  - Confirm DIV_VAR name and grid/coordinate names by inspecting the
    NetCDF header (ds.info()) -- do not assume, per your own convention
    in process_era5_sst_sector.py.
  - Confirm the NSIDC polar stereographic projection parameters against
    the file's grid_mapping / crs metadata rather than trusting the
    approximate cartopy CRS below.
  - Fill in WIND_SOURCE once decided (reuse compute_gridded_sic_wind_diff.py
    if it already loads a gridded wind field).
"""

import glob
import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

# ---------------- CONFIG ----------------
DIVERGENCE_NC = "ice_divergence_daily_sh.nc"
DIV_VAR = "div"            # VERIFY against ds.data_vars
DIV_TIME_COORD = "time"    # VERIFY

SST_RAW_DIR = "era5_sst_raw"
SST_RAW_GLOB = "era5_sst_*.nc"
SST_VAR = "sst"
SST_LAT = "latitude"
SST_LON = "longitude"
SST_TIME = "valid_time"    # matches process_era5_sst_sector.py

WIND_SOURCE = None         # TODO: point at gridded wind stress product

SPLIT_YEAR = 2016
EXCLUDE_YEARS = [1978, 1987, 1991, 1995]   # matches Ch2 / existing scripts

OUT_DIR = "spatial_diagnostics"

# Approximate NSIDC-style south polar stereographic projection.
# VERIFY against the actual grid_mapping attributes in the divergence file
# before trusting map geometry for anything beyond a quick look.
NSIDC_CRS = ccrs.Stereographic(
    central_latitude=-90.0,
    central_longitude=0.0,
    true_scale_latitude=-70.0,
)
PLATE_CARREE = ccrs.PlateCarree()
# -----------------------------------------


def load_divergence():
    ds = xr.open_dataset(DIVERGENCE_NC)
    print("Divergence file variables:", list(ds.data_vars))
    print("Divergence file coords:", list(ds.coords))
    if DIV_VAR not in ds:
        raise KeyError(f"{DIV_VAR!r} not found. Available: {list(ds.data_vars)}")
    return ds


def load_sst_stack():
    files = sorted(glob.glob(os.path.join(SST_RAW_DIR, SST_RAW_GLOB)))
    if not files:
        raise FileNotFoundError(f"No files matched {SST_RAW_GLOB} in {SST_RAW_DIR}")
    parts = [xr.open_dataset(f) for f in files]
    ds = xr.concat(parts, dim=SST_TIME)
    print("SST file variables:", list(ds.data_vars))
    return ds


def add_period_month(ds, time_coord):
    time = pd.to_datetime(ds[time_coord].values)
    year = xr.DataArray(time.year, dims=time_coord, coords={time_coord: ds[time_coord]})
    month = xr.DataArray(time.month, dims=time_coord, coords={time_coord: ds[time_coord]})
    ds = ds.assign_coords(year=year, month=month)
    ds = ds.sel({time_coord: ~ds["year"].isin(EXCLUDE_YEARS)})
    ds = ds.assign_coords(post=(ds["year"] >= SPLIT_YEAR).astype(int))
    return ds


def monthly_prepost_maps(ds, var, time_coord, out_prefix):
    """For each calendar month: pre-2016 mean, post-2016 mean, difference,
    and a per-cell Welch's t-test screen using YEARLY monthly means as the
    sample unit (n = n_years per cell), not daily values -- daily values are
    heavily autocorrelated and would make the per-cell test badly
    overconfident. This still isn't the formal test (that's the sector-level
    block bootstrap); it's a spatial screen only.
    """
    os.makedirs(OUT_DIR, exist_ok=True)

    # collapse to one value per (year, month, cell) first
    yearly_monthly = ds[var].groupby([ds["year"], ds["month"]]).mean(dim=time_coord)

    for m in range(1, 13):
        try:
            month_data = yearly_monthly.sel(month=m)
        except KeyError:
            print(f"[skip] month {m}: no data")
            continue

        years = month_data["year"].values
        post_mask = years >= SPLIT_YEAR
        pre = month_data.isel(year=~post_mask)
        post = month_data.isel(year=post_mask)

        if pre.sizes.get("year", 0) < 3 or post.sizes.get("year", 0) < 3:
            print(f"[skip] month {m}: insufficient years pre/post "
                  f"({pre.sizes.get('year', 0)}/{post.sizes.get('year', 0)})")
            continue

        pre_mean = pre.mean(dim="year")
        post_mean = post.mean(dim="year")
        diff = post_mean - pre_mean

        # per-cell Welch's t-test on yearly monthly means (vectorized)
        tstat, pval = stats.ttest_ind(
            post.values, pre.values, axis=0, equal_var=False, nan_policy="omit"
        )

        _plot_prepost_panel(
            pre_mean, post_mean, diff, pval,
            title=f"{out_prefix} — month {m:02d}",
            out_path=os.path.join(OUT_DIR, f"{out_prefix}_month{m:02d}.png"),
        )
        print(f"  wrote month {m:02d}")


def _plot_prepost_panel(pre_mean, post_mean, diff, pval, title, out_path):
    fig, axes = plt.subplots(
        1, 3, figsize=(15, 5), subplot_kw={"projection": NSIDC_CRS}
    )
    for ax in axes:
        ax.add_feature(cfeature.LAND, zorder=1, facecolor="lightgray")
        ax.coastlines(resolution="50m")
        ax.set_extent([-180, 180, -90, -50], crs=PLATE_CARREE)

    for ax, field, label in zip(
        axes, [pre_mean, post_mean, diff], [f"pre-{SPLIT_YEAR}", f"post-{SPLIT_YEAR}", "post - pre"]
    ):
        vmax = np.nanpercentile(np.abs(field.values), 95)
        im = field.plot.pcolormesh(
            ax=ax, transform=NSIDC_CRS, add_colorbar=False,
            cmap="RdBu_r", vmin=-vmax, vmax=vmax,
        )
        ax.set_title(label)
        fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.8)

    # stipple where diff is NOT significant at p<0.05 -- i.e. highlight the
    # opposite of the usual convention, since here we care where a screened
    # signal DOES vs does NOT show up, and most cells will likely be null
    sig = pval < 0.05
    axes[2].contourf(
        diff["x"], diff["y"], sig, levels=[0.5, 1.5], hatches=["..."],
        colors="none", transform=NSIDC_CRS,
    )

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def sst_coverage_maps(ds):
    """Valid-cell fraction per month -- direct visual on the winter
    sea-ice-masking problem flagged in process_era5_sst_sector.py, rather
    than inferring it from a coverage-threshold table.
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    valid = ds[SST_VAR].notnull()
    monthly_frac = valid.groupby(ds["month"]).mean(dim=SST_TIME)

    fig, axes = plt.subplots(
        3, 4, figsize=(18, 12), subplot_kw={"projection": NSIDC_CRS}
    )
    for m, ax in zip(range(1, 13), axes.flat):
        field = monthly_frac.sel(month=m)
        im = field.plot.pcolormesh(
            ax=ax, transform=PLATE_CARREE, add_colorbar=False,
            cmap="viridis", vmin=0, vmax=1,
        )
        ax.add_feature(cfeature.LAND, zorder=1, facecolor="lightgray")
        ax.coastlines(resolution="50m")
        ax.set_extent([-180, 180, -90, -50], crs=PLATE_CARREE)
        ax.set_title(f"month {m:02d}")

    fig.suptitle("SST valid-cell fraction by month (low = ice-masked)")
    fig.colorbar(im, ax=axes, orientation="horizontal", pad=0.02, shrink=0.6)
    fig.savefig(os.path.join(OUT_DIR, "sst_coverage_by_month.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote sst_coverage_by_month.png")


def main():
    print("=== Divergence ===")
    div_ds = load_divergence()
    div_ds = add_period_month(div_ds, DIV_TIME_COORD)
    monthly_prepost_maps(div_ds, DIV_VAR, DIV_TIME_COORD, "divergence")

    print("\n=== SST ===")
    sst_ds = load_sst_stack()
    sst_ds = add_period_month(sst_ds, SST_TIME)
    monthly_prepost_maps(sst_ds, SST_VAR, SST_TIME, "sst")
    sst_coverage_maps(sst_ds)

    if WIND_SOURCE:
        print("\n=== Wind ===  [TODO: not yet wired up]")
    else:
        print("\n[info] WIND_SOURCE not set; skipping wind maps. "
              "Point this at whatever compute_gridded_sic_wind_diff.py uses.")


if __name__ == "__main__":
    main()