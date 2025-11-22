#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fig_FS_MS_anomaly_maps_static_vs_dynamic.py

Mean spatial anomalies of FS, MS, and season duration
for STATIC vs DYNAMIC methods, in a 2 x 3 panel:

    Row 1: Dynamic  (FS, MS, season duration)
    Row 2: Static   (FS, MS, season duration)

By default this plots the mean anomaly over 2017–2023
relative to the 1980–2016 climatology used when the
anomaly fields were constructed.

Input (already precomputed anomalies):
    results/anomalies/FS_dynamic_anomalies.nc   [FS_dynamic_anom(year,y,x)]
    results/anomalies/MS_dynamic_anomalies.nc   [MS_dynamic_anom(year,y,x)]
    results/anomalies/FS_static_anomalies.nc    [FS_static_anom(year,y,x)]
    results/anomalies/MS_static_anomalies.nc    [MS_static_anom(year,y,x)]

Output:
    results/Ch2_Figures/anomalies/Fig_FS_MS_anomaly_maps_static_vs_dynamic.png
    and mirrored to gdrive:sea-ice-phase/Results/Ch2_Figures/anomalies
"""

import sys
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ---------------------------------------------------------------------
# Import shared Ch2 utilities
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

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

# Period for "post-2016" anomalies (change if you like)
YEAR_POST_START = 2017
YEAR_POST_END   = 2023

# Color range for anomaly maps (symmetric around 0)
VMAX = 30.0  # days

REMOTE_ROOT = "gdrive:sea-ice-phase/Results/Ch2_Figures"
SUBFOLDER   = "anomalies"


# ---------------------------------------------------------------------
# LOAD ANOMALIES
# ---------------------------------------------------------------------

def load_anom_file(fname: str, varname: str) -> xr.DataArray:
    """Load anomaly DataArray (year,y,x)."""
    path = ANOM_DIR / fname
    ds = xr.open_dataset(path)
    if varname not in ds:
        raise KeyError(
            f"{varname} not found in {path.name}; "
            f"available: {list(ds.data_vars)}"
        )
    da = ds[varname]
    ds.close()
    return da


def load_all_anomalies():
    """
    Returns dict with keys:
        FS_dynamic, MS_dynamic, FS_static, MS_static
    Each is a DataArray (year,y,x).
    """
    fs_dyn = load_anom_file("FS_dynamic_anomalies.nc", "FS_dynamic_anom")
    ms_dyn = load_anom_file("MS_dynamic_anomalies.nc", "MS_dynamic_anom")
    fs_sta = load_anom_file("FS_static_anomalies.nc",  "FS_static_anom")
    ms_sta = load_anom_file("MS_static_anomalies.nc",  "MS_static_anom")

    return {
        "FS_dynamic": fs_dyn,
        "MS_dynamic": ms_dyn,
        "FS_static":  fs_sta,
        "MS_static":  ms_sta,
    }


# ---------------------------------------------------------------------
# PERIOD MEANS & DURATION
# ---------------------------------------------------------------------

def mean_over_period(da: xr.DataArray, year_start: int, year_end: int) -> xr.DataArray:
    """Mean anomaly over [year_start, year_end] inclusive."""
    return da.sel(year=slice(year_start, year_end)).mean("year", skipna=True)


def compute_period_means(year_start: int, year_end: int):
    """
    Compute mean anomalies for FS, MS, and season duration
    for dynamic and static methods.

    Returns dict:
      {
        "FS_dynamic":  <DataArray>,
        "MS_dynamic":  <DataArray>,
        "DUR_dynamic": <DataArray>,
        "FS_static":   <DataArray>,
        "MS_static":   <DataArray>,
        "DUR_static":  <DataArray>,
      }
    """
    anoms = load_all_anomalies()

    fs_dyn_mean = mean_over_period(anoms["FS_dynamic"], year_start, year_end)
    ms_dyn_mean = mean_over_period(anoms["MS_dynamic"], year_start, year_end)
    fs_sta_mean = mean_over_period(anoms["FS_static"],  year_start, year_end)
    ms_sta_mean = mean_over_period(anoms["MS_static"],  year_start, year_end)

    # Duration anomaly = MS_anom - FS_anom
    dur_dyn_mean = ms_dyn_mean - fs_dyn_mean
    dur_sta_mean = ms_sta_mean - fs_sta_mean

    return {
        "FS_dynamic":  fs_dyn_mean,
        "MS_dynamic":  ms_dyn_mean,
        "DUR_dynamic": dur_dyn_mean,
        "FS_static":   fs_sta_mean,
        "MS_static":   ms_sta_mean,
        "DUR_static":  dur_sta_mean,
    }


# ---------------------------------------------------------------------
# MAP PLOTTING HELPERS
# ---------------------------------------------------------------------

def make_clean_polar_ax(fig, nrows: int, ncols: int, index: int) -> plt.Axes:
    """
    South polar stereographic axes with:
      - white background
      - grey continents
      - NO coastlines
      - no gridlines (clean look)
    """
    proj = ccrs.SouthPolarStereo()
    ax = fig.add_subplot(nrows, ncols, index, projection=proj)

    # show only Antarctic region
    ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())

    # continents only (grey)
    ax.add_feature(
        cfeature.LAND,
        facecolor="0.8",
        edgecolor="none",
        zorder=2,
    )

    # No ocean feature, no coastlines, no gridlines
    ax.outline_patch.set_visible(False)

    return ax


def pcolormesh_from_da(ax: plt.Axes, da: xr.DataArray, vlim: float, cmap: str = "RdBu_r"):
    """
    Plot DataArray on SouthPolarStereo native grid (x,y).
    Assumes da has coords 'x' and 'y'.
    """
    proj = ccrs.SouthPolarStereo()

    if not {"x", "y"} <= set(da.coords):
        raise ValueError("Anomaly DataArray must have 'x' and 'y' coordinates.")

    x = da["x"]
    y = da["y"]

    im = ax.pcolormesh(
        x,
        y,
        da,
        transform=proj,
        cmap=cmap,
        vmin=-vlim,
        vmax=+vlim,
        shading="auto",
        zorder=1,
    )
    return im


# ---------------------------------------------------------------------
# MAIN FIGURE
# ---------------------------------------------------------------------

def plot_anomaly_maps():
    # Compute period means
    period_means = compute_period_means(YEAR_POST_START, YEAR_POST_END)

    fs_dyn = period_means["FS_dynamic"]
    ms_dyn = period_means["MS_dynamic"]
    dur_dyn = period_means["DUR_dynamic"]

    fs_sta = period_means["FS_static"]
    ms_sta = period_means["MS_static"]
    dur_sta = period_means["DUR_static"]

    # Figure: 2 rows (dynamic, static) x 3 columns (FS, MS, duration)
    fig = plt.figure(figsize=(7.5, 6.0))  # sized for a Word doc page
    fig.patch.set_facecolor("white")

    titles_row1 = ["Dynamic FS", "Dynamic MS", "Dynamic season duration"]
    titles_row2 = ["Static FS",  "Static MS",  "Static season duration"]

    data_row1 = [fs_dyn, ms_dyn, dur_dyn]
    data_row2 = [fs_sta, ms_sta, dur_sta]

    all_axes = []
    all_ims = []

    # Row 1: dynamic
    for i, (da, title) in enumerate(zip(data_row1, titles_row1), start=1):
        ax = make_clean_polar_ax(fig, nrows=2, ncols=3, index=i)
        im = pcolormesh_from_da(ax, da, vlim=VMAX)
        ax.set_title(title, fontsize=9, fontweight="bold")
        all_axes.append(ax)
        all_ims.append(im)

    # Row 2: static
    for j, (da, title) in enumerate(zip(data_row2, titles_row2), start=4):
        ax = make_clean_polar_ax(fig, nrows=2, ncols=3, index=j)
        im = pcolormesh_from_da(ax, da, vlim=VMAX)
        ax.set_title(title, fontsize=9, fontweight="bold")
        all_axes.append(ax)
        all_ims.append(im)

    # Shared colorbar
    # Use the last im; vmin/vmax are the same for all
    cax = fig.add_axes([0.15, 0.08, 0.7, 0.03])
    cb = fig.colorbar(all_ims[-1], cax=cax, orientation="horizontal")
    cb.set_label(
        f"Mean anomaly (days, {YEAR_POST_START}–{YEAR_POST_END} relative to 1980–2016)",
        fontsize=9,
    )
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_visible(False)

    fig.suptitle(
        "FS, MS, and season duration anomalies: static vs dynamic",
        fontsize=11,
        fontweight="bold",
        y=0.97,
    )

    fig.tight_layout(rect=[0.02, 0.13, 0.98, 0.95])

    out_path = get_fig_path(
        project_root=PROJECT_ROOT,
        subfolder=SUBFOLDER,
        fig_name="Fig_FS_MS_anomaly_maps_static_vs_dynamic.png",
    )

    save_and_upload(
        fig,
        out_path,
        remote_root=REMOTE_ROOT,
        remote_subdir=SUBFOLDER,
    )


def main():
    plot_anomaly_maps()


if __name__ == "__main__":
    main()
