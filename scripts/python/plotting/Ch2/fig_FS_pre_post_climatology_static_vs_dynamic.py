#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fig_FS_pre_post_climatology_static_vs_dynamic.py

Advance (FS) climatologies, dynamic vs static:

Rows:
  top   = dynamic FS
  bottom= static FS

Columns:
  1: 1980–2016 mean FS
  2: 2017–2023 mean FS
  3: Post − Pre (2017–2023 minus 1980–2016)

Grey continent, no coastlines, white background.
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
PLOTTING_ROOT = HERE.parent  # .../plotting

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

YEAR_PRE_START = 1980
YEAR_PRE_END   = 2016
YEAR_POST_START = 2017
YEAR_POST_END   = 2023  # whatever max year you actually have

# colour limits – adjust if needed
FS_VMIN = 80.0
FS_VMAX = 240.0
DIFF_LIM = 30.0  # ± days

# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------

def _load_fs_dyn():
    ds = xr.open_dataset(FS_DYN_CLIM_FILE)
    da = ds["FS_dynamic_clim"]  # (year, y, x)
    return da


def _load_fs_sta():
    ds = xr.open_dataset(FS_STA_CLIM_FILE)
    da = ds["FS_static_clim"]  # (year, y, x)
    return da


def _pre_post_means(da: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    da: (year, y, x)
    Returns pre, post, diff (post - pre).
    """
    da = ds[var]  # already time-collapsed
    pre = da.sel(year=slice(YEAR_PRE_START, YEAR_PRE_END)).mean("year", skipna=True)
    post = da.sel(year=slice(YEAR_POST_START, YEAR_POST_END)).mean("year", skipna=True)
    diff = post - pre
    return pre, post, diff


def make_clean_polar_ax(fig, nrows, ncols, index):
    """
    South polar stereographic axes, grey land, no coastlines.
    """
    proj = ccrs.SouthPolarStereo()
    ax = fig.add_subplot(nrows, ncols, index, projection=proj)

    ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="0.75", edgecolor="none", zorder=1)
    ax.add_feature(cfeature.OCEAN, facecolor="white", edgecolor="none", zorder=0)
    # no coastlines, no gridlines
    ax.set_facecolor("white")
    return ax


# ---------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------

def plot_fs_pre_post_maps():
    fs_dyn = _load_fs_dyn()
    fs_sta = _load_fs_sta()

    # assume same x/y grid for dynamic + static
    x = fs_dyn["x"]
    y = fs_dyn["y"]
    proj = ccrs.SouthPolarStereo()

    fs_dyn_pre, fs_dyn_post, fs_dyn_diff = _pre_post_means(fs_dyn)
    fs_sta_pre, fs_sta_post, fs_sta_diff = _pre_post_means(fs_sta)

    fig = plt.figure(figsize=(10, 7))
    fig.patch.set_facecolor("white")

    # order: row 1 = dynamic, row 2 = static
    panels = [
        ("Dynamic, 1980–2016", fs_dyn_pre, FS_VMIN, FS_VMAX, "phase"),
        ("Dynamic, 2017–2023", fs_dyn_post, FS_VMIN, FS_VMAX, "phase"),
        ("Dynamic, Post − Pre", fs_dyn_diff, -DIFF_LIM, DIFF_LIM, "diff"),
        ("Static, 1980–2016", fs_sta_pre, FS_VMIN, FS_VMAX, "phase"),
        ("Static, 2017–2023", fs_sta_post, FS_VMIN, FS_VMAX, "phase"),
        ("Static, Post − Pre", fs_sta_diff, -DIFF_LIM, DIFF_LIM, "diff"),
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

    # phase colorbar (bottom left)
    cax1 = fig.add_axes([0.13, 0.10, 0.32, 0.03])
    cb1 = fig.colorbar(
        ims[0][0],
        cax=cax1,
        orientation="horizontal",
    )
    cb1.set_label("Freeze start date (day of year)", fontsize=9)
    cb1.ax.tick_params(labelsize=8)
    cb1.outline.set_visible(False)

    # diff colorbar (bottom right)
    cax2 = fig.add_axes([0.55, 0.10, 0.32, 0.03])
    cb2 = fig.colorbar(
        ims[2][0],
        cax=cax2,
        orientation="horizontal",
    )
    cb2.set_label("Δ freeze start (Post − Pre, days)", fontsize=9)
    cb2.ax.tick_params(labelsize=8)
    cb2.outline.set_visible(False)

    fig.suptitle(
        "Advance (FS) climatologies: 1980–2016 vs 2017–2023 (dynamic vs static)",
        fontsize=12,
        fontweight="bold",
    )

    fig.tight_layout(rect=[0.02, 0.16, 0.98, 0.93])

    out_path = get_fig_path(
        project_root=PROJECT_ROOT,
        subfolder="anomalies",
        fig_name="Fig_FS_pre_post_climatology_static_vs_dynamic.png",
    )

    save_and_upload(
        fig,
        out_path,
        remote_root="gdrive:sea-ice-phase/Results/Ch2_Figures",
        remote_subdir="anomalies",
    )


def main():
    plot_fs_pre_post_maps()


if __name__ == "__main__":
    main()
