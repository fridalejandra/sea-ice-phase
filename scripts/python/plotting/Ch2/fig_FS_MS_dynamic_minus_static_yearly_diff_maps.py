#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fig_FS_MS_dynamic_minus_static_yearly_diff_maps.py

For each year, reconstruct FS and MS fields for static and dynamic
from the stored climatology + anomaly files, then plot:

    ΔFS = FS_dynamic  - FS_static
    ΔMS = MS_dynamic  - MS_static

as 2-panel polar maps (one figure per year).

This is purely a diagnostic to see how the dynamic method differs
from the static method on a year-by-year basis, without climatological
smoothing.

Input (existing files; already created in your workflow):

    results/anomalies/FS_dynamic_climatology.nc   (var: FS_dynamic_clim)
    results/anomalies/FS_dynamic_anomalies.nc     (var: FS_dynamic_anom)
    results/anomalies/FS_static_climatology.nc    (var: FS_static_clim)
    results/anomalies/FS_static_anomalies.nc      (var: FS_static_anom)

    results/anomalies/MS_dynamic_climatology.nc   (var: MS_dynamic_clim)
    results/anomalies/MS_dynamic_anomalies.nc     (var: MS_dynamic_anom)
    results/anomalies/MS_static_climatology.nc    (var: MS_static_clim)
    results/anomalies/MS_static_anomalies.nc      (var: MS_static_anom)

Mask / sectors:

    data/canonical_sectors.nc   (var: valid_ocean)

Output:

    One PNG per year, e.g.
      results/Ch2_Figures/diagnostics/yearly_FS_MS_dynamic_minus_static/
          FS_MS_dyn_minus_static_YYYY.png

(and mirrored to gdrive under the same subfolder).
"""

import sys
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ---------------------------------------------------------------------
# Import ch2_fig_utils from parent plotting directory
# ---------------------------------------------------------------------
HERE = Path(__file__).resolve().parent          # .../plotting/Ch2
PLOTTING_ROOT = HERE.parent                     # .../plotting

if str(PLOTTING_ROOT) not in sys.path:
    sys.path.insert(0, str(PLOTTING_ROOT))

from ch2_fig_utils import (  # noqa: E402
    set_mpl_defaults,
    get_fig_path,
    save_and_upload,
    PROJECT_ROOT_CLUSTER,
)

set_mpl_defaults()

PROJECT_ROOT = PROJECT_ROOT_CLUSTER
ANOM_DIR = PROJECT_ROOT / "results" / "anomalies"
SECTOR_FILE = PROJECT_ROOT / "data" / "canonical_sectors.nc"

# Years you want to loop over
YEAR_MIN = 1980
YEAR_MAX = 2023
YEARS = np.arange(YEAR_MIN, YEAR_MAX + 1)

# Diverging range for ΔFS / ΔMS (days)
DMAX = 30.0  # symmetric ±DMAX


# ---------------------------------------------------------------------
# Load climatologies & anomalies once
# ---------------------------------------------------------------------
def load_all():
    fs_dyn_clim = xr.open_dataset(
        ANOM_DIR / "FS_dynamic_climatology.nc"
    )["FS_dynamic_clim"]
    fs_dyn_anom = xr.open_dataset(
        ANOM_DIR / "FS_dynamic_anomalies.nc"
    )["FS_dynamic_anom"]

    fs_stat_clim = xr.open_dataset(
        ANOM_DIR / "FS_static_climatology.nc"
    )["FS_static_clim"]
    fs_stat_anom = xr.open_dataset(
        ANOM_DIR / "FS_static_anomalies.nc"
    )["FS_static_anom"]

    ms_dyn_clim = xr.open_dataset(
        ANOM_DIR / "MS_dynamic_climatology.nc"
    )["MS_dynamic_clim"]
    ms_dyn_anom = xr.open_dataset(
        ANOM_DIR / "MS_dynamic_anomalies.nc"
    )["MS_dynamic_anom"]

    ms_stat_clim = xr.open_dataset(
        ANOM_DIR / "MS_static_climatology.nc"
    )["MS_static_clim"]
    ms_stat_anom = xr.open_dataset(
        ANOM_DIR / "MS_static_anomalies.nc"
    )["MS_static_anom"]

    # Sector mask (valid ocean)
    mask_ds = xr.open_dataset(SECTOR_FILE)
    valid_ocean = mask_ds["valid_ocean"].astype(bool)

    return (fs_dyn_clim, fs_dyn_anom,
            fs_stat_clim, fs_stat_anom,
            ms_dyn_clim, ms_dyn_anom,
            ms_stat_clim, ms_stat_anom,
            valid_ocean)


# ---------------------------------------------------------------------
# Map helper
# ---------------------------------------------------------------------
def make_clean_polar_ax(fig, position):
    """
    Create a clean South Polar stereographic axis with grey land,
    no coastlines, no ticks/labels.
    """
    proj = ccrs.SouthPolarStereo()
    ax = fig.add_subplot(position, projection=proj)

    ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
    ax.add_feature(
        cfeature.LAND,
        facecolor="0.8",
        edgecolor="0.8",
        zorder=1,
    )
    # No coastlines, no grid, no frame
    ax.gridlines(draw_labels=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax


# ---------------------------------------------------------------------
# Main plotting loop
# ---------------------------------------------------------------------
def plot_yearly_diff_maps():
    (
        fs_dyn_clim, fs_dyn_anom,
        fs_stat_clim, fs_stat_anom,
        ms_dyn_clim, ms_dyn_anom,
        ms_stat_clim, ms_stat_anom,
        valid_ocean,
    ) = load_all()

    # Use x,y from one of the fields (they should all share the grid)
    x = fs_dyn_clim["x"]
    y = fs_dyn_clim["y"]
    proj = ccrs.SouthPolarStereo()

    for year in YEARS:
        if year not in fs_dyn_anom["year"].values:
            # skip if year outside anomaly range
            continue

        # Reconstruct absolute FS/MS dates for this year
        fs_dyn = (fs_dyn_clim + fs_dyn_anom.sel(year=year))
        fs_stat = (fs_stat_clim + fs_stat_anom.sel(year=year))

        ms_dyn = (ms_dyn_clim + ms_dyn_anom.sel(year=year))
        ms_stat = (ms_stat_clim + ms_stat_anom.sel(year=year))

        # Δ (dynamic - static)
        dfs = (fs_dyn - fs_stat).where(valid_ocean)
        dms = (ms_dyn - ms_stat).where(valid_ocean)

        # Figure
        fig = plt.figure(figsize=(7.5, 4.0))
        fig.patch.set_facecolor("white")

        # ΔFS panel
        ax1 = make_clean_polar_ax(fig, 121)
        im1 = ax1.pcolormesh(
            x,
            y,
            dfs,
            transform=proj,
            cmap="RdBu_r",
            vmin=-DMAX,
            vmax=+DMAX,
            shading="auto",
        )
        ax1.set_title(f"ΔFS (dynamic − static), {year}", fontsize=9)

        # ΔMS panel
        ax2 = make_clean_polar_ax(fig, 122)
        im2 = ax2.pcolormesh(
            x,
            y,
            dms,
            transform=proj,
            cmap="RdBu_r",
            vmin=-DMAX,
            vmax=+DMAX,
            shading="auto",
        )
        ax2.set_title(f"ΔMS (dynamic − static), {year}", fontsize=9)

        # Shared colorbar
        cax = fig.add_axes([0.15, 0.08, 0.7, 0.04])
        cb = fig.colorbar(im2, cax=cax, orientation="horizontal")
        cb.set_label("Difference in date (days)", fontsize=9)
        cb.ax.tick_params(labelsize=8)
        cb.outline.set_visible(False)

        fig.suptitle(
            "Yearly phase differences: dynamic vs static",
            fontsize=10,
            fontweight="bold",
        )

        fig.tight_layout(rect=[0, 0.13, 1, 0.93])

        # Save locally + to gdrive
        fig_name = f"FS_MS_dyn_minus_static_{year}.png"
        out_path = get_fig_path(
            project_root=PROJECT_ROOT,
            subfolder="diagnostics/yearly_FS_MS_dynamic_minus_static",
            fig_name=fig_name,
        )

        save_and_upload(
            fig,
            out_path,
            remote_root="gdrive:sea-ice-phase/Results/Ch2_Figures",
            remote_subdir="diagnostics/yearly_FS_MS_dynamic_minus_static",
        )

        plt.close(fig)
        print(f"Saved {fig_name}")


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
def main():
    plot_yearly_diff_maps()


if __name__ == "__main__":
    main()
