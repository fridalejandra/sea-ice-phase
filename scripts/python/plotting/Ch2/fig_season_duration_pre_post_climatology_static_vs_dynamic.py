#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fig_season_duration_pre_post_climatology_static_vs_dynamic.py

Season duration climatologies (MS − FS), dynamic vs static:

Rows:
  top   = dynamic duration (MS_dyn - FS_dyn)
  bottom= static duration (MS_sta - FS_sta)

Columns:
  1: 1980–2016 mean
  2: 2017–2023 mean
  3: Post − Pre
"""

import sys
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ---------------------------------------------------------------------
# ch2_fig_utils + paths
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

FS_DYN_CLIM_FILE = ANOM_DIR / "FS_dynamic_climatology.nc"
FS_STA_CLIM_FILE = ANOM_DIR / "FS_static_climatology.nc"
MS_DYN_CLIM_FILE = ANOM_DIR / "MS_dynamic_climatology.nc"
MS_STA_CLIM_FILE = ANOM_DIR / "MS_static_climatology.nc"

YEAR_PRE_START = 1980
YEAR_PRE_END   = 2016
YEAR_POST_START = 2017
YEAR_POST_END   = 2023

DUR_VMIN = 150.0  # rough range of duration (days) – tweak if needed
DUR_VMAX = 300.0
DIFF_LIM = 30.0   # ± days

# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------

def _load_dyn_duration():
    fs = xr.open_dataset(FS_DYN_CLIM_FILE)["FS_dynamic_clim"]
    ms = xr.open_dataset(MS_DYN_CLIM_FILE)["MS_dynamic_clim"]
    # season duration = MS - FS (both DOY)
    dur = ms - fs
    return dur


def _load_sta_duration():
    fs = xr.open_dataset(FS_STA_CLIM_FILE)["FS_static_clim"]
    ms = xr.open_dataset(MS_STA_CLIM_FILE)["MS_static_clim"]
    dur = ms - fs
    return dur


def _pre_post_means(da: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    da = da.sortby("year")
    pre = da.sel(year=slice(YEAR_PRE_START, YEAR_PRE_END)).mean("year", skipna=True)
    post = da.sel(year=slice(YEAR_POST_START, YEAR_POST_END)).mean("year", skipna=True)
    diff = post - pre
    return pre, post, diff


def make_clean_polar_ax(fig, nrows, ncols, index):
    proj = ccrs.SouthPolarStereo()
    ax = fig.add_subplot(nrows, ncols, index, projection=proj)
    ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="0.75", edgecolor="none", zorder=1)
    ax.add_feature(cfeature.OCEAN, facecolor="white", edgecolor="none", zorder=0)
    ax.set_facecolor("white")
    return ax


# ---------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------

def plot_duration_pre_post_maps():
    dur_dyn = _load_dyn_duration()
    dur_sta = _load_sta_duration()

    x = dur_dyn["x"]
    y = dur_dyn["y"]
    proj = ccrs.SouthPolarStereo()

    dur_dyn_pre, dur_dyn_post, dur_dyn_diff = _pre_post_means(dur_dyn)
    dur_sta_pre, dur_sta_post, dur_sta_diff = _pre_post_means(dur_sta)

    fig = plt.figure(figsize=(10, 7))
    fig.patch.set_facecolor("white")

    panels = [
        ("Dynamic, 1980–2016", dur_dyn_pre, DUR_VMIN, DUR_VMAX, "phase"),
        ("Dynamic, 2017–2023", dur_dyn_post, DUR_VMIN, DUR_VMAX, "phase"),
        ("Dynamic, Post − Pre", dur_dyn_diff, -DIFF_LIM, DIFF_LIM, "diff"),
        ("Static, 1980–2016", dur_sta_pre, DUR_VMIN, DUR_VMAX, "phase"),
        ("Static, 2017–2023", dur_sta_post, DUR_VMIN, DUR_VMAX, "phase"),
        ("Static, Post − Pre", dur_sta_diff, -DIFF_LIM, DIFF_LIM, "diff"),
    ]

    ims = []
    for i, (title, field, vmin, vmax, kind) in enumerate(panels, start=1):
        ax = make_clean_polar_ax(fig, nrows=2, ncols=3, index=i)
        cmap = "viridis" if kind == "phase" else "RdBu_r"
        im = ax.pcolormesh(
            x,
            y,
            field,
            transform=proj,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="auto",
        )
        ims.append((im, kind))
        ax.set_title(title, fontsize=10, fontweight="bold")

    # duration colorbar
    cax1 = fig.add_axes([0.13, 0.10, 0.32, 0.03])
    cb1 = fig.colorbar(ims[0][0], cax=cax1, orientation="horizontal")
    cb1.set_label("Season duration (days)", fontsize=9)
    cb1.ax.tick_params(labelsize=8)
    cb1.outline.set_visible(False)

    # diff colorbar
    cax2 = fig.add_axes([0.55, 0.10, 0.32, 0.03])
    cb2 = fig.colorbar(ims[2][0], cax=cax2, orientation="horizontal")
    cb2.set_label("Δ season duration (Post − Pre, days)", fontsize=9)
    cb2.ax.tick_params(labelsize=8)
    cb2.outline.set_visible(False)

    fig.suptitle(
        "Season duration climatologies: 1980–2016 vs 2017–2023 (dynamic vs static)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0.02, 0.16, 0.98, 0.93])

    out_path = get_fig_path(
        project_root=PROJECT_ROOT,
        subfolder="anomalies",
        fig_name="Fig_season_duration_pre_post_climatology_static_vs_dynamic.png",
    )

    save_and_upload(
        fig,
        out_path,
        remote_root="gdrive:sea-ice-phase/Results/Ch2_Figures",
        remote_subdir="anomalies",
    )


def main():
    plot_duration_pre_post_maps()


if __name__ == "__main__":
    main()
