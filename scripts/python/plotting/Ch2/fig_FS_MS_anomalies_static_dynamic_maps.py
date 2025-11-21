#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fig_FS_MS_anomalies_static_dynamic_maps.py

Spatial anomalies of FS and MS (static + dynamic) for selected years.

Uses precomputed anomalies:

  results/anomalies/FS_static_anomalies.nc      (FS_static_anom[year,y,x])
  results/anomalies/FS_dynamic_anomalies.nc     (FS_dynamic_anom[year,y,x])
  results/anomalies/MS_static_anomalies.nc      (MS_static_anom[year,y,x])
  results/anomalies/MS_dynamic_anomalies.nc     (MS_dynamic_anom[year,y,x])

For each target year, produces a 2x3 panel:

  (a) FS static anomaly (days)
  (b) FS dynamic anomaly (days)
  (c) FS (dynamic − static) anomaly (days)
  (d) MS static anomaly (days)
  (e) MS dynamic anomaly (days)
  (f) MS (dynamic − static) anomaly (days)

Map style:
  - South polar stereographic
  - Grey continents, no coastlines, white ocean
  - white figure background
"""

import sys
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ---------------------------------------------------------------------
# ch2_fig_utils import
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

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
ANOM_DIR = PROJECT_ROOT_CLUSTER / "results" / "anomalies"

# years of interest
TARGET_YEARS = [2016, 2022, 2023]

# anomaly files and variable names (from the compute_FS_MS_anomalies script)
ANOM_FILES = {
    ("FS", "static"):  ("FS_static_anomalies.nc",  "FS_static_anom"),
    ("FS", "dynamic"): ("FS_dynamic_anomalies.nc", "FS_dynamic_anom"),
    ("MS", "static"):  ("MS_static_anomalies.nc",  "MS_static_anom"),
    ("MS", "dynamic"): ("MS_dynamic_anomalies.nc", "MS_dynamic_anom"),
}

# colour scale (days)
VMAX = 40.0  # symmetric ±VMAX


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def load_anom_da(phase: str, method: str) -> xr.DataArray:
    """
    Load anomaly DataArray for given phase ('FS' or 'MS') and method
    ('static' or 'dynamic') with dims (year, y, x).
    """
    fname, varname = ANOM_FILES[(phase, method)]
    fpath = ANOM_DIR / fname
    if not fpath.exists():
        raise FileNotFoundError(f"Missing anomaly file: {fpath}")

    ds = xr.open_dataset(fpath)
    if varname not in ds:
        raise KeyError(
            f"Variable {varname} not in {fpath}. "
            f"Available: {list(ds.data_vars)}"
        )
    da = ds[varname]
    ds.close()
    return da


def get_xy_from_da(da: xr.DataArray):
    """Extract x,y coords from anomaly DataArray."""
    if "x" not in da.coords or "y" not in da.coords:
        raise ValueError("Anomaly DataArray is missing x/y coordinates.")
    return da["x"], da["y"]


def make_polar_ax(fig, pos):
    """
    South polar stereographic axes with:
      - grey continents
      - no coastlines
      - white ocean/background
    """
    proj = ccrs.SouthPolarStereo()
    ax = fig.add_subplot(pos, projection=proj)

    ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())

    # White background
    ax.set_facecolor("white")

    # Grey continents, no coastlines
    ax.add_feature(
        cfeature.LAND.with_scale("110m"),
        facecolor="0.85",
        edgecolor="none",
        zorder=1,
    )

    # No gridlines/meridians
    gl = ax.gridlines(draw_labels=False)
    gl.xlines = False
    gl.ylines = False

    return ax


def plot_anomaly_maps_for_year(year: int):
    """
    Build a 2x3 panel for a single year.
    """
    print(f"\n=== Plotting anomalies for {year} ===")

    # Load full anomaly fields
    fs_stat_all = load_anom_da("FS", "static")
    fs_dyn_all  = load_anom_da("FS", "dynamic")
    ms_stat_all = load_anom_da("MS", "static")
    ms_dyn_all  = load_anom_da("MS", "dynamic")

    # Check year is available
    if year not in fs_stat_all["year"].values:
        raise ValueError(f"Year {year} not found in FS static anomalies.")
    if year not in fs_dyn_all["year"].values:
        raise ValueError(f"Year {year} not found in FS dynamic anomalies.")
    if year not in ms_stat_all["year"].values:
        raise ValueError(f"Year {year} not found in MS static anomalies.")
    if year not in ms_dyn_all["year"].values:
        raise ValueError(f"Year {year} not found in MS dynamic anomalies.")

    # Select the year slice (y,x)
    fs_stat = fs_stat_all.sel(year=year)
    fs_dyn  = fs_dyn_all.sel(year=year)
    ms_stat = ms_stat_all.sel(year=year)
    ms_dyn  = ms_dyn_all.sel(year=year)

    # Difference anomalies (dynamic − static)
    fs_diff = fs_dyn - fs_stat
    ms_diff = ms_dyn - ms_stat

    # x,y from one field
    x, y = get_xy_from_da(fs_stat)

    proj = ccrs.SouthPolarStereo()

    # Sized for Word: ~7" wide, moderate height
    fig = plt.figure(figsize=(7.2, 5.0))
    fig.patch.set_facecolor("white")

    panels = [
        ("(a) FS static anomaly",   fs_stat, 231),
        ("(b) FS dynamic anomaly",  fs_dyn,  232),
        ("(c) FS dyn − static",     fs_diff, 233),
        ("(d) MS static anomaly",   ms_stat, 234),
        ("(e) MS dynamic anomaly",  ms_dyn,  235),
        ("(f) MS dyn − static",     ms_diff, 236),
    ]

    last_im = None
    for title, da, code in panels:
        ax = make_polar_ax(fig, code)

        im = ax.pcolormesh(
            x,
            y,
            da,
            transform=proj,
            cmap="RdBu_r",
            vmin=-VMAX,
            vmax=+VMAX,
            shading="auto",
        )
        last_im = im

        ax.set_title(title, fontsize=9, fontweight="bold")

    # Shared colorbar at bottom
    cax = fig.add_axes([0.15, 0.08, 0.7, 0.03])
    cb = fig.colorbar(last_im, cax=cax, orientation="horizontal")
    cb.set_label("Anomaly (days)", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_visible(False)

    fig.suptitle(
        f"FS/MS anomalies (static vs dynamic) for {year}",
        fontsize=10,
        fontweight="bold",
        y=0.98,
    )

    fig.subplots_adjust(left=0.04, right=0.96, top=0.93, bottom=0.14, wspace=0.12, hspace=0.15)

    # Save
    out_path = get_fig_path(
        PROJECT_ROOT_CLUSTER,
        subfolder="anomalies/spatial",
        fig_name=f"Fig_FS_MS_anomalies_static_dynamic_{year}.png",
    )

    save_and_upload(
        fig,
        out_path,
        remote_root="gdrive:sea-ice-phase/Results/Ch2_Figures",
        remote_subdir="anomalies/spatial",
    )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    for year in TARGET_YEARS:
        plot_anomaly_maps_for_year(year)


if __name__ == "__main__":
    main()
