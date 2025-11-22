#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fig_MS_pre_post_climatology_static_vs_dynamic.py

Retreat (MS) pre/post climatology maps:

  - Pre:  1980–2016 mean retreat date
  - Post: 2017–2023 mean retreat date
  - Δ:    Post − Pre (days)

Rows:
  top    = dynamic method
  bottom = static method

Uses previously saved anomalies / climatology files:

  results/anomalies/MS_dynamic_climatology.nc
  results/anomalies/MS_dynamic_anomalies.nc
  results/anomalies/MS_static_climatology.nc
  results/anomalies/MS_static_anomalies.nc
"""

import sys
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ---------------------------------------------------------------------
# Import shared Ch2 helpers
# ---------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
PLOTTING_ROOT = HERE.parent

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

# pre/post periods
PRE_START, PRE_END = 1980, 2016
POST_START, POST_END = 2017, 2023  # adjust to 2024 when anomalies exist

# plotting limits
DOY_VMIN, DOY_VMAX = 260, 340   # retreat-ish; tweak if needed
DV_VLIM = 30                    # ±30 days for difference


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def load_ms_dyn():
    ds_clim = xr.open_dataset(ANOM_DIR / "MS_dynamic_climatology.nc")
    ds_anom = xr.open_dataset(ANOM_DIR / "MS_dynamic_anomalies.nc")
    clim = ds_clim["MS_dynamic_clim"]
    anom = ds_anom["MS_dynamic_anom"]
    return clim, anom

def load_ms_static():
    ds_clim = xr.open_dataset(ANOM_DIR / "MS_static_climatology.nc")
    ds_anom = xr.open_dataset(ANOM_DIR / "MS_static_anomalies.nc")
    clim = ds_clim["MS_static_clim"]
    anom = ds_anom["MS_static_anom"]
    return clim, anom
w

def pre_post_from_clim_anom(clim: xr.DataArray, anom: xr.DataArray):
    """
    Given a baseline climatology (1980–2016) and annual anomalies relative
    to that baseline, return:

      pre_clim  : DOY (y,x)  (1980–2016 mean)   -> just `clim`
      post_clim : DOY (y,x)  (2017–post_end mean)
      diff      : days (y,x) (post − pre)      -> mean anomaly post period
    """
    years = anom["year"].values
    # Guard in case POST_END extends beyond available years
    mask_post = (years >= POST_START) & (years <= POST_END)
    if not mask_post.any():
        raise ValueError("No post-period years found in anomalies dataset.")

    anom_post = anom.sel(year=years[mask_post]).mean("year", skipna=True)

    pre_clim = clim
    post_clim = clim + anom_post
    diff = anom_post  # post − pre

    return pre_clim, post_clim, diff


def make_clean_polar_ax(fig, nrows, ncols, index):
    proj = ccrs.SouthPolarStereo()
    ax = fig.add_subplot(nrows, ncols, index, projection=proj)

    ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="0.8", edgecolor="0.8", zorder=1)
    ax.set_facecolor("white")
    # No coastlines, no gridlines
    return ax


# ---------------------------------------------------------------------
# Main plotting routine
# ---------------------------------------------------------------------
def plot_ms_pre_post_maps():
    # --- dynamic
    ms_dyn_clim, ms_dyn_anom = load_ms_dyn()
    ms_dyn_pre, ms_dyn_post, ms_dyn_diff = pre_post_from_clim_anom(
        ms_dyn_clim, ms_dyn_anom
    )

    # --- static
    ms_stat_clim, ms_stat_anom = load_ms_static()
    ms_stat_pre, ms_stat_post, ms_stat_diff = pre_post_from_clim_anom(
        ms_stat_clim, ms_stat_anom
    )

    # coordinates (assume x,y stereo grid)
    x = ms_dyn_clim["x"]
    y = ms_dyn_clim["y"]
    proj = ccrs.SouthPolarStereo()

    fig = plt.figure(figsize=(9, 6))
    fig.patch.set_facecolor("white")

    panels = [
        ("Dynamic, 1980–2016", ms_dyn_pre, 1, "clim"),
        ("Dynamic, 2017–2023", ms_dyn_post, 2, "clim"),
        ("Dynamic, Post – Pre", ms_dyn_diff, 3, "diff"),
        ("Static, 1980–2016", ms_stat_pre, 4, "clim"),
        ("Static, 2017–2023", ms_stat_post, 5, "clim"),
        ("Static, Post – Pre", ms_stat_diff, 6, "diff"),
    ]

    im_clim_handles = []
    im_diff_handles = []

    for title, field, idx, kind in panels:
        ax = make_clean_polar_ax(fig, nrows=2, ncols=3, index=idx)

        if kind == "clim":
            im = ax.pcolormesh(
                x,
                y,
                field,
                cmap="viridis",
                vmin=DOY_VMIN,
                vmax=DOY_VMAX,
                transform=proj,
                shading="auto",
            )
            im_clim_handles.append(im)
        else:  # diff
            im = ax.pcolormesh(
                x,
                y,
                field,
                cmap="RdBu_r",
                vmin=-DV_VLIM,
                vmax=+DV_VLIM,
                transform=proj,
                shading="auto",
            )
            im_diff_handles.append(im)

        ax.set_title(title, fontsize=9, fontweight="bold")

    # shared colorbars: one for climatologies, one for difference
    # climatology cbar under left/middle panels
    cax1 = fig.add_axes([0.15, 0.08, 0.35, 0.03])
    cb1 = fig.colorbar(im_clim_handles[0], cax=cax1, orientation="horizontal")
    cb1.set_label("Retreat date (day of year)", fontsize=9)
    cb1.ax.tick_params(labelsize=8)
    cb1.outline.set_visible(False)

    # difference cbar under right panels
    cax2 = fig.add_axes([0.55, 0.08, 0.30, 0.03])
    cb2 = fig.colorbar(im_diff_handles[0], cax=cax2, orientation="horizontal")
    cb2.set_label("Δ retreat (Post − Pre, days)", fontsize=9)
    cb2.ax.tick_params(labelsize=8)
    cb2.outline.set_visible(False)

    fig.suptitle(
        "Retreat (MS) climatologies: 1980–2016 vs 2017–2023 (dynamic vs static)",
        fontsize=11,
        fontweight="bold",
    )

    fig.tight_layout(rect=[0, 0.14, 1, 0.95])

    out_path = get_fig_path(
        PROJECT_ROOT,
        subfolder="anomalies",
        fig_name="Fig_MS_pre_post_climatology_static_vs_dynamic.png",
    )

    save_and_upload(
        fig,
        out_path,
        remote_root="gdrive:sea-ice-phase/Results/Ch2_Figures",
        remote_subdir="anomalies",
    )


def main():
    plot_ms_pre_post_maps()


if __name__ == "__main__":
    main()
