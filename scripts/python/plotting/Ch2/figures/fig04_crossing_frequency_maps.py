#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig04_crossing_frequency_maps.py

Climatological mean threshold-crossing frequency during the seasonal window,
for Freeze Start (FS) and Melt Start (MS), SMMR 1979-2023.

Panels:
  (a) FS mean crossing frequency (climatology)
  (b) MS mean crossing frequency (climatology)

These maps show where SIC repeatedly crosses the 15% threshold during the
seasonal window — i.e. where synoptic forcing causes flickering that a fixed
persistence window may detect as a genuine seasonal transition. High crossing
frequency = high sensitivity to persistence window choice.

Inputs:
  data/transition_metrics/SMMR/crossing_freq_FS_thr15.nc  (year, y, x)
  data/transition_metrics/SMMR/crossing_freq_MS_thr15.nc  (year, y, x)

Output:
  results/Ch2_Figures/Fig04_crossing_frequency_FS_MS_SMMR_thr15_clim.png
"""

import sys
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ---------------------------------------------------------------------
# project root on sys.path
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[5]
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
SENSOR      = "SMMR"
THRESH_PCT  = 15
YEAR_MIN    = 1979
YEAR_MAX    = 2023
METRICS_DIR = PROJECT_ROOT / "data" / "transition_metrics" / SENSOR
REMOTE_ROOT = "gdrive:sea-ice-phase/results/Ch2_Figures"
SUBFOLDER   = ""
VMAX        = 6.0   # crossings — p95 is 6, cap there


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def load_crossing_freq(phase: str) -> xr.DataArray:
    """
    Load crossing_freq (year, y, x) for given phase and threshold,
    subset to YEAR_MIN–YEAR_MAX.
    """
    path = METRICS_DIR / f"crossing_freq_{phase}_thr{THRESH_PCT}.nc"
    ds   = xr.open_dataset(path)
    cf   = ds["crossing_freq"].sel(year=slice(YEAR_MIN, YEAR_MAX))
    ds.close()
    return cf


def make_polar_ax(fig, pos):
    proj = ccrs.SouthPolarStereo()
    ax   = fig.add_subplot(pos, projection=proj)
    ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
    ax.add_feature(
        cfeature.LAND.with_scale("110m"),
        facecolor="0.85",
        edgecolor="0.6",
        linewidth=0.4,
        zorder=3,
    )
    ax.coastlines(linewidth=0.4, color="0.4", zorder=4)
    gl = ax.gridlines(draw_labels=False)
    gl.xlines = False
    gl.ylines = False
    ax.set_facecolor("white")
    return ax


# ---------------------------------------------------------------------
# MAIN PLOT
# ---------------------------------------------------------------------

def plot_crossing_freq_maps() -> None:
    set_mpl_defaults()

    print("Loading crossing frequency data...")
    cf_fs = load_crossing_freq("FS")
    cf_ms = load_crossing_freq("MS")

    # climatological mean over years
    clim_fs = cf_fs.mean("year", skipna=True)
    clim_ms = cf_ms.mean("year", skipna=True)

    print(f"FS clim — mean: {float(clim_fs.mean(skipna=True)):.2f}  "
          f"p95: {float(clim_fs.quantile(0.95, skipna=True)):.2f}  "
          f"max: {float(clim_fs.max(skipna=True)):.2f}")
    print(f"MS clim — mean: {float(clim_ms.mean(skipna=True)):.2f}  "
          f"p95: {float(clim_ms.quantile(0.95, skipna=True)):.2f}  "
          f"max: {float(clim_ms.max(skipna=True)):.2f}")

    x = clim_fs["x"].values
    y = clim_fs["y"].values
    proj = ccrs.SouthPolarStereo()

    cmap = plt.cm.YlOrRd
    norm = mcolors.Normalize(vmin=0, vmax=VMAX)

    fig = plt.figure(figsize=(10.0, 5.0))

    panels = [
        ("(a) FS — freeze onset window", clim_fs, 121),
        ("(b) MS — melt onset window",   clim_ms, 122),
    ]

    axes   = []
    im_last = None
    for title, da, subplot_code in panels:
        ax = make_polar_ax(fig, subplot_code)
        axes.append(ax)
        im_last = ax.pcolormesh(
            x, y, da.values,
            transform=proj,
            cmap=cmap,
            norm=norm,
            shading="auto",
            zorder=1,
        )
        ax.set_title(title, fontsize=10, pad=4)

    # shared colorbar
    cax = fig.add_axes([0.2, 0.06, 0.6, 0.03])
    cb  = fig.colorbar(im_last, cax=cax, orientation="horizontal", extend="max")
    cb.set_label(
        f"Mean threshold crossings per season ({YEAR_MIN}–{YEAR_MAX})",
        fontsize=9
    )
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_visible(False)

    fig.suptitle(
        f"Climatological crossing frequency — {SENSOR} thr={THRESH_PCT}%",
        fontsize=11, y=0.97
    )
    fig.tight_layout(rect=[0, 0.12, 1, 0.95])

    fig_name = format_fig_name(
        num=4,
        short=f"crossing_frequency_FS_MS_{SENSOR}_thr{THRESH_PCT}_clim",
    )
    out_path = get_fig_path(
        project_root=PROJECT_ROOT,
        subfolder=SUBFOLDER,
        fig_name=fig_name,
    )
    save_and_upload(
        fig, out_path,
        remote_root=REMOTE_ROOT,
        remote_subdir=SUBFOLDER,
    )


if __name__ == "__main__":
    plot_crossing_freq_maps()