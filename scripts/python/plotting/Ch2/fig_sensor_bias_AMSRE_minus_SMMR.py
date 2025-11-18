#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare AMSRE vs SMMR phase dates (old static advance/retreat)
and produce:
  - yearly wrapped-bias maps (AMSRE - SMMR)
  - climatological wrapped-bias map
  - histogram of all wrapped biases

Uses ch2_fig_utils for style, naming, paths, and rclone upload.
"""

from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import sys
from pathlib import Path

# Add project root to sys.path so "scripts.*" imports work
PROJECT_ROOT = Path(__file__).resolve().parents[4]
# parents[4] should be .../sea-ice-phase
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts.python.plotting.ch2_fig_utils import (
    set_mpl_defaults,
    format_fig_name,
    get_fig_path,
    save_and_upload,
)

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

set_mpl_defaults()

# cluster repo root
PROJECT_ROOT = Path("/user/geog/falejandraperez/sea-ice-phase")

# rclone remote for figures
REMOTE_ROOT = "gdrive:sea-ice-phase/Results/Ch2_Figures"

# phase name in the old files: "advance" or "retreat"
PHASE = "retreat"

YEARS = range(2012, 2024)   # AMSRE era you care about

# where the per-year phase files live
SMMR_DIR  = PROJECT_ROOT / "results" / "SMMR_phase"
AMSRE_DIR = PROJECT_ROOT / "results" / "AMSRE_phase"

# output subfolder under Results/Ch2_Figures
SUBFOLDER = f"sensor/{PHASE}"


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def wrapped_difference(a, b, period=365):
    """
    Circular difference: (a - b) in [-period/2, period/2]
    e.g. day-of-year differences, so 360 vs 5 becomes -10, not +355.
    """
    diff = (a - b + period / 2) % period - period / 2
    return diff


def plot_bias_map(
    bias_da,
    title,
    vlim=20,
):
    """
    Simple South Polar Stereo map of bias in days.
    bias_da: xarray.DataArray with dims (y, x) and lon/lat coords or SMMR grid.
    """
    data = bias_da.values
    # If lon/lat coords exist, use them; otherwise assume SMMR grid (x,y)
    if {"lon", "lat"} <= set(bias_da.coords):
        lons = bias_da["lon"].values
        lats = bias_da["lat"].values
    else:
        # fall back: try to get from separate file later if needed
        raise ValueError("bias_da must have lon/lat coords")

    proj = ccrs.SouthPolarStereo()
    data_crs = ccrs.PlateCarree()

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(1, 1, 1, projection=proj)

    ax.set_extent([-180, 180, -90, -50], crs=data_crs)
    ax.add_feature(cfeature.OCEAN, facecolor="black", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="0.7", edgecolor="0.7", zorder=1)
    ax.coastlines(linewidth=0.4, zorder=2)

    im = ax.pcolormesh(
        lons,
        lats,
        data,
        transform=data_crs,
        cmap="RdBu_r",
        vmin=-vlim,
        vmax=+vlim,
    )

    cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.8)
    cb.set_label(f"{PHASE.capitalize()} bias (AMSRE − SMMR, days)")

    ax.set_title(title)
    return fig, ax


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    all_bias_flat = []

    # load one SMMR file to get lon/lat grid
    sample_ds = xr.open_dataset(SMMR_DIR / f"seaice_phases_SMMR_{YEARS.start}.nc")
    lons = sample_ds["lon"]
    lats = sample_ds["lat"]

    yearly_bias_list = []

    for year in YEARS:
        print(f"Processing year {year} ({PHASE})")

        varname = f"{PHASE}_{year}"   # old static naming

        smmr_path  = SMMR_DIR  / f"seaice_phases_SMMR_{year}.nc"
        amsre_path = AMSRE_DIR / f"seaice_phases_AMSRE_{year}.nc"

        ds_smmr  = xr.open_dataset(smmr_path)
        ds_amsre = xr.open_dataset(amsre_path)

        smmr = ds_smmr[varname].load()        # [y_smmr, x_smmr]
        amsr = ds_amsre[varname].load()       # [y_amsr, x_amsr] (finer grid)

        # coarsen AMSR to SMMR resolution (2x2 averaging)
        ny, nx = amsr.shape
        if ny % 2 != 0:
            amsr = amsr.isel(y=slice(0, ny - 1))
        if nx % 2 != 0:
            amsr = amsr.isel(x=slice(0, nx - 1))

        amsr_coarse = amsr.coarsen(y=2, x=2, boundary="trim").mean()

        # sanity: align dimensions with SMMR
        amsr_coarse = amsr_coarse.assign_coords(x=smmr.x, y=smmr.y)

        bias_vals = wrapped_difference(amsr_coarse.values, smmr.values, period=365.0)

        bias = xr.DataArray(
            bias_vals,
            coords={"y": smmr.y, "x": smmr.x, "lon": lons, "lat": lats},
            dims=("y", "x"),
            name="bias",
        )

        yearly_bias_list.append(bias)

        # accumulate for histogram
        all_bias_flat.append(bias.values.ravel())

        # ---- yearly bias map (appendix-type) ----
        title = f"{PHASE.capitalize()} wrapped bias (AMSRE − SMMR), {year}"
        fig_year, ax_year = plot_bias_map(bias, title, vlim=20)

        yearly_out = get_fig_path(
            project_root=PROJECT_ROOT,
            subfolder=f"{SUBFOLDER}/yearly",
            fig_name=f"sensor_{PHASE}_bias_AMSREminusSMMR_{year}.png",
        )

        save_and_upload(
            fig_year,
            yearly_out,
            remote_root=REMOTE_ROOT,
            remote_subdir=f"{SUBFOLDER}/yearly",
        )

    # -----------------------------------------------------------------
    # Climatological bias (mean over years)
    # -----------------------------------------------------------------
    bias_stack = xr.concat(yearly_bias_list, dim="year")
    clim_bias = bias_stack.mean("year", skipna=True)

    fig_clim, ax_clim = plot_bias_map(
        clim_bias,
        title=f"{PHASE.capitalize()} wrapped bias (AMSRE − SMMR), {YEARS.start}–{YEARS.stop - 1}",
        vlim=20,
    )

    clim_name = format_fig_name(
        num=7,  # <-- adjust to whatever figure number this will be in the chapter
        short=f"sensor_{PHASE}_bias_AMSREminusSMMR_{YEARS.start}-{YEARS.stop - 1}",
    )

    clim_out = get_fig_path(
        project_root=PROJECT_ROOT,
        subfolder=SUBFOLDER,
        fig_name=clim_name,
    )

    save_and_upload(
        fig_clim,
        clim_out,
        remote_root=REMOTE_ROOT,
        remote_subdir=SUBFOLDER,
    )

    # -----------------------------------------------------------------
    # Histogram of wrapped bias
    # -----------------------------------------------------------------
    all_bias = np.concatenate(all_bias_flat)
    all_bias = all_bias[~np.isnan(all_bias)]

    fig_h, ax_h = plt.subplots(figsize=(4, 3))
    ax_h.hist(all_bias, bins=np.arange(-40, 41, 2), density=True, alpha=0.8, edgecolor="none")
    ax_h.axvline(0, color="k", linewidth=0.8)
    ax_h.set_xlabel(f"{PHASE.capitalize()} bias (AMSRE − SMMR, days)")
    ax_h.set_ylabel("Probability density")
    ax_h.set_title(f"{PHASE.capitalize()} wrapped bias distribution\n{YEARS.start}–{YEARS.stop - 1}")

    hist_name = format_fig_name(
        num=8,  # next figure number, adjust as needed
        short=f"sensor_{PHASE}_bias_histogram_AMSREminusSMMR_{YEARS.start}-{YEARS.stop - 1}",
    )

    hist_out = get_fig_path(
        project_root=PROJECT_ROOT,
        subfolder=SUBFOLDER,
        fig_name=hist_name,
    )

    save_and_upload(
        fig_h,
        hist_out,
        remote_root=REMOTE_ROOT,
        remote_subdir=SUBFOLDER,
    )


if __name__ == "__main__":
    main()
