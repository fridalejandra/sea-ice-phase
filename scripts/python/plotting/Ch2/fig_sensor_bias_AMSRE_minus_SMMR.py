#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fig_sensor_bias_AMSRE_minus_SMMR.py

Compare AMSRE vs SMMR phase dates (old static advance/retreat)
and produce:

  - climatological wrapped-bias map (AMSRE − SMMR, days)
  - histogram of that wrapped bias

Uses ch2_fig_utils for style, naming, paths, and rclone upload.
"""

import sys
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ---------------------------------------------------------------------
# Make sure project root is on sys.path so "scripts.*" imports work
#   this assumes this file lives in:
#   sea-ice-phase/scripts/python/plotting/Ch2/
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[4]  # -> sea-ice-phase
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.python.plotting.ch2_fig_utils import (  # noqa: E402
    set_mpl_defaults,
    format_fig_name,
    get_fig_path,
    save_and_upload,
)

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

# phase variable in the old static files: "advance" or "retreat"
PHASE = "retreat"

# AMSRE period you care about (years that exist for BOTH AMSRE and SMMR)
YEARS = range(2012, 2024)  # 2012–2023 inclusive

# where the per-year phase files live (adjust AMSRE_DIR if needed)
SMMR_DIR = PROJECT_ROOT / "results" / "SMMR_phase"
AMSRE_DIR = PROJECT_ROOT / "results" / "AMSRE_phase"

# where to put figures (relative to PROJECT_ROOT and your rclone remote)
SUBFOLDER = f"sensor/{PHASE}"
REMOTE_ROOT = "gdrive:sea-ice-phase/Results/Ch2_Figures"


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def wrapped_difference(a, b, period=365.0):
    """
    Circular difference: (a - b) mapped into [-period/2, period/2].

    Example: 360 vs 5 becomes -10, not +355.
    """
    diff = (a - b + period / 2.0) % period - period / 2.0
    return diff


def plot_bias_map(bias_da, title, vlim=20):
    """
    South Polar Stereo map of bias in days, using native x/y grid
    (no lon/lat in these older phase files).
    """
    data = bias_da.values
    x = bias_da["x"].values
    y = bias_da["y"].values

    proj = ccrs.SouthPolarStereo()
    fig, ax = plt.subplots(
        figsize=(6, 6),
        subplot_kw={"projection": proj},
    )

    im = ax.pcolormesh(
        x,
        y,
        data,
        transform=proj,          # data already in this projection
        cmap="RdBu_r",
        vmin=-vlim,
        vmax=+vlim,
        shading="auto",
    )

    # just use a standard Antarctic extent for context
    ax.set_extent([-180, 180, -90, -50], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN, facecolor="black", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="0.7", edgecolor="0.7", zorder=1)
    ax.coastlines(linewidth=0.4, zorder=2)

    cbar = fig.colorbar(
        im,
        ax=ax,
        orientation="horizontal",
        pad=0.05,
        shrink=0.8,
    )
    cbar.set_label(f"{PHASE.capitalize()} bias (AMSRE − SMMR, days)")

    ax.set_title(title)
    fig.tight_layout()

    return fig, ax


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    set_mpl_defaults()

    all_bias_flat = []

    yearly_bias_list = []

    for year in YEARS:
        print(f"Processing {PHASE} for {year}")

        # old static variable names are like "advance_2012", "retreat_2012"
        varname = f"{PHASE}_{year}"

        smmr_path = SMMR_DIR / f"seaice_phases_SMMR_{year}.nc"
        amsre_path = AMSRE_DIR / f"seaice_phases_AMSRE_{year}.nc"

        ds_smmr = xr.open_dataset(smmr_path)
        ds_amsre = xr.open_dataset(amsre_path)

        smmr = ds_smmr[varname].load()   # [y_smmr, x_smmr]
        amsr = ds_amsre[varname].load()  # [y_amsr, x_amsr] on finer grid

        # coarsen AMSRE to SMMR resolution (assumes dims are "y", "x")
        ny, nx = amsr.shape
        if ny % 2 != 0:
            amsr = amsr.isel(y=slice(0, ny - 1))
        if nx % 2 != 0:
            amsr = amsr.isel(x=slice(0, nx - 1))

        amsr_coarse = amsr.coarsen(y=2, x=2, boundary="trim").mean()

        # align coordinates with SMMR grid
        amsr_coarse = amsr_coarse.assign_coords(x=smmr.x, y=smmr.y)

        # wrapped bias in days
        bias_vals = wrapped_difference(amsr_coarse.values, smmr.values, period=365.0)

        bias = xr.DataArray(
            data=bias_vals,
            coords={"y": smmr.y, "x": smmr.x},
            dims=("y", "x"),
            name="bias",
        )

        yearly_bias_list.append(bias)
        all_bias_flat.append(bias.values.ravel())

    # -----------------------------------------------------------------
    # Climatological bias (mean over years)
    # -----------------------------------------------------------------
    bias_stack = xr.concat(yearly_bias_list, dim="year")
    bias_clim = bias_stack.mean("year", skipna=True)

    title_clim = (
        f"{PHASE.capitalize()} wrapped bias (AMSRE − SMMR), "
        f"{YEARS.start}–{YEARS.stop - 1}"
    )
    fig_clim, ax_clim = plot_bias_map(bias_clim, title=title_clim, vlim=20)

    clim_name = format_fig_name(
        num=7,  # adjust to your figure numbering
        short=f"sensor_{PHASE}_bias_AMSREminusSMMR_{YEARS.start}-{YEARS.stop - 1}",
    )

    clim_path = get_fig_path(
        project_root=PROJECT_ROOT,
        subfolder=SUBFOLDER,
        fig_name=clim_name,
    )

    save_and_upload(
        fig_clim,
        clim_path,
        remote_root=REMOTE_ROOT,
        remote_subdir=SUBFOLDER,
    )

    # -----------------------------------------------------------------
    # Histogram of wrapped bias
    # -----------------------------------------------------------------
    all_bias = np.concatenate(all_bias_flat)
    all_bias = all_bias[~np.isnan(all_bias)]

    fig_h, ax_h = plt.subplots(figsize=(4, 3))
    ax_h.hist(
        all_bias,
        bins=np.arange(-40, 42, 2),
        density=True,
        alpha=0.8,
        edgecolor="none",
    )
    ax_h.axvline(0, color="k", linewidth=0.8)
    ax_h.set_xlabel(f"{PHASE.capitalize()} bias (AMSRE − SMMR, days)")
    ax_h.set_ylabel("Probability density")
    ax_h.set_title(
        f"{PHASE.capitalize()} wrapped bias distribution\n"
        f"{YEARS.start}–{YEARS.stop - 1}"
    )

    hist_name = format_fig_name(
        num=8,  # next figure number
        short=f"sensor_{PHASE}_bias_hist_AMSREminusSMMR_{YEARS.start}-{YEARS.stop - 1}",
    )

    hist_path = get_fig_path(
        project_root=PROJECT_ROOT,
        subfolder=SUBFOLDER,
        fig_name=hist_name,
    )

    save_and_upload(
        fig_h,
        hist_path,
        remote_root=REMOTE_ROOT,
        remote_subdir=SUBFOLDER,
    )


if __name__ == "__main__":
    main()
