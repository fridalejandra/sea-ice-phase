#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Single-year FS/MS anomaly maps: static vs dynamic.

Panels (for a chosen year):

(a) FS static anomaly      = FS_static_anom(year)
(b) FS dynamic anomaly     = FS_dynamic_anom(year)
(c) FS dyn − static        = (FS_dynamic_anom − FS_static_anom)

(d) MS static anomaly      = MS_static_anom(year)
(e) MS dynamic anomaly     = MS_dynamic_anom(year)
(f) MS dyn − static        = (MS_dynamic_anom − MS_static_anom)

All in days, relative to their own method-specific climatologies.
"""

import sys
from pathlib import Path

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

from utils.plot_utils import (  # noqa: E402
    set_mpl_defaults,
    get_fig_path,
    save_and_upload,
    PROJECT_ROOT_CLUSTER,
)

set_mpl_defaults()

PROJECT_ROOT = PROJECT_ROOT_CLUSTER
ANOM_DIR = PROJECT_ROOT / "data" / "anomalies" / "SMMR"
SECTOR_FILE = PROJECT_ROOT / "data" / "canonical_sectors.nc"

REMOTE_ROOT = "gdrive:sea-ice-phase/results/Ch2_Figures"
SUBFOLDER   = ""

# year to plot
TARGET_YEAR = 2015  # change as needed

VMAX = 40.0  # days, symmetric range for anomalies/diffs


# ---------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------
def load_fs_ms_clim_anom():
    """Load FS/MS climatology & anomalies for static and dynamic + valid_ocean."""
    fs_dyn_clim = xr.open_dataset(ANOM_DIR / "FS_dynamic_k5_q70_climatology.nc", decode_times=False)["FS_dynamic_k5_q70_clim"]
    fs_dyn_anom = xr.open_dataset(ANOM_DIR / "FS_dynamic_k5_q70_anomalies.nc", decode_times=False)["FS_dynamic_k5_q70_anom"]

    ms_dyn_clim = xr.open_dataset(ANOM_DIR / "MS_dynamic_k5_q70_climatology.nc", decode_times=False)["MS_dynamic_k5_q70_clim"]
    ms_dyn_anom = xr.open_dataset(ANOM_DIR / "MS_dynamic_k5_q70_anomalies.nc", decode_times=False)["MS_dynamic_k5_q70_anom"]

    fs_sta_clim = xr.open_dataset(ANOM_DIR / "FS_static_thr15_k5_climatology.nc", decode_times=False)["FS_static_thr15_k5_clim"]
    fs_sta_anom = xr.open_dataset(ANOM_DIR / "FS_static_thr15_k5_anomalies.nc", decode_times=False)["FS_static_thr15_k5_anom"]

    ms_sta_clim = xr.open_dataset(ANOM_DIR / "MS_static_thr15_k5_climatology.nc", decode_times=False)["MS_static_thr15_k5_clim"]
    ms_sta_anom = xr.open_dataset(ANOM_DIR / "MS_static_thr15_k5_anomalies.nc", decode_times=False)["MS_static_thr15_k5_anom"]

    try:
        ds_mask = xr.open_dataset(SECTOR_FILE)
        valid_ocean = ds_mask["valid_ocean"].astype(bool)
        ds_mask.close()
    except FileNotFoundError:
        valid_ocean = None

    return {
        "FS_dynamic_anom": fs_dyn_anom,
        "MS_dynamic_anom": ms_dyn_anom,
        "FS_static_anom":  fs_sta_anom,
        "MS_static_anom":  ms_sta_anom,
        "valid_ocean":     valid_ocean,
    }


def make_clean_polar_ax(fig, nrows, ncols, index):
    proj = ccrs.SouthPolarStereo()
    ax = fig.add_subplot(nrows, ncols, index, projection=proj)
    ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="0.8", edgecolor="none", zorder=2)
    ax.set_facecolor("white")
    ax.gridlines(draw_labels=False, linewidth=0.3, color="0.7", alpha=0.4, linestyle="--")
    return ax, proj


def plot_panel(ax, da, proj, vlim, title):
    if not {"x", "y"} <= set(da.coords):
        raise ValueError("DataArray must have 'x' and 'y' coordinates.")
    x = da["x"]
    y = da["y"]

    im = ax.pcolormesh(
        x,
        y,
        da,
        transform=proj,
        cmap="RdBu_r",
        vmin=-vlim,
        vmax=vlim,
        shading="auto",
    )
    ax.set_title(title, fontsize=9, fontweight="bold")
    return im


# ---------------------------------------------------------------------
# Main plotting
# ---------------------------------------------------------------------
def make_single_year_maps(year: int, fields: dict) -> None:
    fs_dyn_anom = fields["FS_dynamic_anom"].sel(year=year)
    ms_dyn_anom = fields["MS_dynamic_anom"].sel(year=year)
    fs_sta_anom = fields["FS_static_anom"].sel(year=year)
    ms_sta_anom = fields["MS_static_anom"].sel(year=year)
    valid_ocean = fields["valid_ocean"]

    # dynamic − static anomalies
    fs_diff = fs_dyn_anom - fs_sta_anom
    ms_diff = ms_dyn_anom - ms_sta_anom

    if valid_ocean is not None:
        fs_dyn_anom = fs_dyn_anom.where(valid_ocean)
        ms_dyn_anom = ms_dyn_anom.where(valid_ocean)
        fs_sta_anom = fs_sta_anom.where(valid_ocean)
        ms_sta_anom = ms_sta_anom.where(valid_ocean)
        fs_diff = fs_diff.where(valid_ocean)
        ms_diff = ms_diff.where(valid_ocean)

    fig = plt.figure(figsize=(9, 6))

    ims = []

    # Row 1: FS
    ax, proj = make_clean_polar_ax(fig, 2, 3, 1)
    ims.append(plot_panel(ax, fs_sta_anom, proj, VMAX, "(a) FS static anomaly"))

    ax, proj = make_clean_polar_ax(fig, 2, 3, 2)
    ims.append(plot_panel(ax, fs_dyn_anom, proj, VMAX, "(b) FS dynamic anomaly"))

    ax, proj = make_clean_polar_ax(fig, 2, 3, 3)
    ims.append(plot_panel(ax, fs_diff, proj, VMAX, "(c) FS dyn − static"))

    # Row 2: MS
    ax, proj = make_clean_polar_ax(fig, 2, 3, 4)
    ims.append(plot_panel(ax, ms_sta_anom, proj, VMAX, "(d) MS static anomaly"))

    ax, proj = make_clean_polar_ax(fig, 2, 3, 5)
    ims.append(plot_panel(ax, ms_dyn_anom, proj, VMAX, "(e) MS dynamic anomaly"))

    ax, proj = make_clean_polar_ax(fig, 2, 3, 6)
    ims.append(plot_panel(ax, ms_diff, proj, VMAX, "(f) MS dyn − static"))

    # Colorbar
    cax = fig.add_axes([0.15, 0.08, 0.7, 0.03])
    cb = fig.colorbar(ims[-1], cax=cax, orientation="horizontal")
    cb.set_label("Anomaly (days)", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_visible(False)

    fig.tight_layout(rect=[0.02, 0.14, 0.98, 0.98])

    fig_name = f"FigS04_FS_MS_anomaly_maps_static_vs_dynamic_{year}.png"
    out_path = get_fig_path(
        project_root=PROJECT_ROOT,
        subfolder=SUBFOLDER,
        fig_name=fig_name,
    )

    save_and_upload(
        fig,
        out_path,
        remote_root=REMOTE_ROOT,
        remote_subdir="",
    )


def main():
    fields = load_fs_ms_clim_anom()
    make_single_year_maps(TARGET_YEAR, fields)


if __name__ == "__main__":
    main()
