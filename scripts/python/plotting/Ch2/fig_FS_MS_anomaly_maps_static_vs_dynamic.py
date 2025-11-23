#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fig_FS_MS_anomaly_maps_static_vs_dynamic.py

FS, MS, and season duration anomalies (post − pre) for
STATIC vs DYNAMIC methods, for multiple pre/post splits:

  A: pre = 1980–2016, post = 2017–2023
  B: pre = 1980–2015, post = 2016–2023
  D: pre = 1980–2017, post = 2018–2023

Each split produces a 2 × 3 panel:

  Row 1: Dynamic  (FS, MS, duration)
  Row 2: Static   (FS, MS, duration)

Values are mean(post) − mean(pre) in days.

Inputs (already produced in your workflow):

  results/anomalies/FS_dynamic_climatology.nc   (FS_dynamic_clim[y,x])
  results/anomalies/FS_dynamic_anomalies.nc     (FS_dynamic_anom[year,y,x])
  results/anomalies/MS_dynamic_climatology.nc   (MS_dynamic_clim[y,x])
  results/anomalies/MS_dynamic_anomalies.nc     (MS_dynamic_anom[year,y,x])

  results/anomalies/FS_static_climatology.nc    (FS_static_clim[y,x])
  results/anomalies/FS_static_anomalies.nc      (FS_static_anom[year,y,x])
  results/anomalies/MS_static_climatology.nc    (MS_static_clim[y,x])
  results/anomalies/MS_static_anomalies.nc      (MS_static_anom[year,y,x])

Optional mask:

  data/canonical_sectors.nc   (valid_ocean[y,x])

Output (one figure per period split), e.g.:

  results/Ch2_Figures/anomalies/
      Fig_FS_MS_anomaly_maps_static_vs_dynamic_pre1980-2016_post2017-2023.png
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
SECTOR_FILE = PROJECT_ROOT / "data" / "canonical_sectors.nc"

REMOTE_ROOT = "gdrive:sea-ice-phase/Results/Ch2_Figures"
SUBFOLDER   = "anomalies"

# ---------------------------------------------------------------------
# Period configs: A, B, D
# ---------------------------------------------------------------------
PERIOD_CONFIGS = {
    "A_pre1980-2016_post2017-2023": {
        "pre_start": 1980,
        "pre_end":   2016,
        "post_start": 2017,
        "post_end":   2023,
    },
    "B_pre1980-2015_post2016-2023": {
        "pre_start": 1980,
        "pre_end":   2015,
        "post_start": 2016,
        "post_end":   2023,
    },
    "D_pre1980-2017_post2018-2023": {
        "pre_start": 1980,
        "pre_end":   2017,
        "post_start": 2018,
        "post_end":   2023,
    },
}

# Amplitude limits (days)
VMAX_DIFF = 30.0   # for FS/MS/duration post−pre maps

# ---------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------

def load_fs_ms_clim_anom():
    """
    Load FS/MS climatology and anomalies for static and dynamic.

    Returns dict with keys:
      FS_dynamic_clim, FS_dynamic_anom,
      MS_dynamic_clim, MS_dynamic_anom,
      FS_static_clim,  FS_static_anom,
      MS_static_clim,  MS_static_anom,
      valid_ocean (mask)
    """
    # dynamic FS/MS
    fs_dyn_clim = xr.open_dataset(
        ANOM_DIR / "FS_dynamic_climatology.nc"
    )["FS_dynamic_clim"]
    fs_dyn_anom = xr.open_dataset(
        ANOM_DIR / "FS_dynamic_anomalies.nc"
    )["FS_dynamic_anom"]

    ms_dyn_clim = xr.open_dataset(
        ANOM_DIR / "MS_dynamic_climatology.nc"
    )["MS_dynamic_clim"]
    ms_dyn_anom = xr.open_dataset(
        ANOM_DIR / "MS_dynamic_anomalies.nc"
    )["MS_dynamic_anom"]

    # static FS/MS
    fs_sta_clim = xr.open_dataset(
        ANOM_DIR / "FS_static_climatology.nc"
    )["FS_static_clim"]
    fs_sta_anom = xr.open_dataset(
        ANOM_DIR / "FS_static_anomalies.nc"
    )["FS_static_anom"]

    ms_sta_clim = xr.open_dataset(
        ANOM_DIR / "MS_static_climatology.nc"
    )["MS_static_clim"]
    ms_sta_anom = xr.open_dataset(
        ANOM_DIR / "MS_static_anomalies.nc"
    )["MS_static_anom"]

    # optional valid ocean mask
    try:
        ds_mask = xr.open_dataset(SECTOR_FILE)
        valid_ocean = ds_mask["valid_ocean"].astype(bool)
    except FileNotFoundError:
        valid_ocean = None

    return {
        "FS_dynamic_clim": fs_dyn_clim,
        "FS_dynamic_anom": fs_dyn_anom,
        "MS_dynamic_clim": ms_dyn_clim,
        "MS_dynamic_anom": ms_dyn_anom,
        "FS_static_clim":  fs_sta_clim,
        "FS_static_anom":  fs_sta_anom,
        "MS_static_clim":  ms_sta_clim,
        "MS_static_anom":  ms_sta_anom,
        "valid_ocean":     valid_ocean,
    }


# ---------------------------------------------------------------------
# Pre/post computation using clim + anom
# ---------------------------------------------------------------------

def compute_pre_post(clim: xr.DataArray,
                     anom: xr.DataArray,
                     pre_start: int,
                     pre_end: int,
                     post_start: int,
                     post_end: int) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Given a climatology (clim[y,x]) and anomalies anom[year,y,x] defined as:

       anom(year) = FS(year) - clim

    we reconstruct mean fields over pre and post periods:

       pre_mean  = mean_{y in pre}( FS(y) )
       post_mean = mean_{y in post}( FS(y) )

    using FS(y) = clim + anom(y).

    Returns:
       pre_mean, post_mean, diff = post_mean - pre_mean
    """
    years = anom["year"].values

    # restrict to years available in anomalies
    pre_mask = (years >= pre_start) & (years <= pre_end)
    post_mask = (years >= post_start) & (years <= post_end)

    if not pre_mask.any():
        raise ValueError(f"No pre years in anomalies for {pre_start}-{pre_end}")
    if not post_mask.any():
        raise ValueError(f"No post years in anomalies for {post_start}-{post_end}")

    anom_pre = anom.sel(year=years[pre_mask]).mean("year", skipna=True)
    anom_post = anom.sel(year=years[post_mask]).mean("year", skipna=True)

    pre_mean = clim + anom_pre
    post_mean = clim + anom_post
    diff = post_mean - pre_mean

    return pre_mean, post_mean, diff


# ---------------------------------------------------------------------
# Map plotting helpers
# ---------------------------------------------------------------------

def make_clean_polar_ax(fig, nrows, ncols, index):
    proj = ccrs.SouthPolarStereo()
    ax = fig.add_subplot(nrows, ncols, index, projection=proj)

    ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
    ax.add_feature(
        cfeature.LAND,
        facecolor="0.8",
        edgecolor="none",
        zorder=2,
    )
    ax.set_facecolor("white")
    # No coastlines, no grid labels
    ax.gridlines(draw_labels=False, linewidth=0.3, color="0.7", alpha=0.4, linestyle="--")
    return ax


def plot_panel(ax, da, proj, vlim, title):
    """
    Plot a single panel with symmetric RdBu_r around 0.
    """
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
        vmax=+vlim,
        shading="auto",
    )
    ax.set_title(title, fontsize=9, fontweight="bold")
    return im


# ---------------------------------------------------------------------
# Main figure generator for one period config
# ---------------------------------------------------------------------

def make_anomaly_map_for_period(period_key: str,
                                cfg: dict,
                                fields: dict) -> None:
    """
    For one pre/post definition, compute post−pre anomalies for FS, MS, duration,
    static and dynamic, and plot a 2×3 panel.
    """
    pre_s = cfg["pre_start"]
    pre_e = cfg["pre_end"]
    post_s = cfg["post_start"]
    post_e = cfg["post_end"]

    fs_dyn_clim = fields["FS_dynamic_clim"]
    fs_dyn_anom = fields["FS_dynamic_anom"]
    ms_dyn_clim = fields["MS_dynamic_clim"]
    ms_dyn_anom = fields["MS_dynamic_anom"]

    fs_sta_clim = fields["FS_static_clim"]
    fs_sta_anom = fields["FS_static_anom"]
    ms_sta_clim = fields["MS_static_clim"]
    ms_sta_anom = fields["MS_static_anom"]

    valid_ocean = fields["valid_ocean"]

    # FS: post - pre
    _, _, fs_dyn_diff = compute_pre_post(fs_dyn_clim, fs_dyn_anom,
                                         pre_s, pre_e, post_s, post_e)
    _, _, fs_sta_diff = compute_pre_post(fs_sta_clim, fs_sta_anom,
                                         pre_s, pre_e, post_s, post_e)

    # MS: post - pre
    _, _, ms_dyn_diff = compute_pre_post(ms_dyn_clim, ms_dyn_anom,
                                         pre_s, pre_e, post_s, post_e)
    _, _, ms_sta_diff = compute_pre_post(ms_sta_clim, ms_sta_anom,
                                         pre_s, pre_e, post_s, post_e)

    # Duration = MS - FS; do pre/post separately then difference
    fs_dyn_pre, fs_dyn_post, _ = compute_pre_post(fs_dyn_clim, fs_dyn_anom,
                                                  pre_s, pre_e, post_s, post_e)
    ms_dyn_pre, ms_dyn_post, _ = compute_pre_post(ms_dyn_clim, ms_dyn_anom,
                                                  pre_s, pre_e, post_s, post_e)

    dur_dyn_pre = ms_dyn_pre - fs_dyn_pre
    dur_dyn_post = ms_dyn_post - fs_dyn_post
    dur_dyn_diff = dur_dyn_post - dur_dyn_pre

    fs_sta_pre, fs_sta_post, _ = compute_pre_post(fs_sta_clim, fs_sta_anom,
                                                  pre_s, pre_e, post_s, post_e)
    ms_sta_pre, ms_sta_post, _ = compute_pre_post(ms_sta_clim, ms_sta_anom,
                                                  pre_s, pre_e, post_s, post_e)

    dur_sta_pre = ms_sta_pre - fs_sta_pre
    dur_sta_post = ms_sta_post - ms_sta_pre
    dur_sta_diff = dur_sta_post - dur_sta_pre

    # Apply valid ocean mask if present
    if valid_ocean is not None:
        fs_dyn_diff = fs_dyn_diff.where(valid_ocean)
        ms_dyn_diff = ms_dyn_diff.where(valid_ocean)
        dur_dyn_diff = dur_dyn_diff.where(valid_ocean)

        fs_sta_diff = fs_sta_diff.where(valid_ocean)
        ms_sta_diff = ms_sta_diff.where(valid_ocean)
        dur_sta_diff = dur_sta_diff.where(valid_ocean)

    # Plot
    proj = ccrs.SouthPolarStereo()
    fig = plt.figure(figsize=(9.0, 6.0))
    fig.patch.set_facecolor("white")

    data_dyn = [
        (fs_dyn_diff, "Dynamic FS"),
        (ms_dyn_diff, "Dynamic MS"),
        (dur_dyn_diff, "Dynamic duration"),
    ]
    data_sta = [
        (fs_sta_diff, "Static FS"),
        (ms_sta_diff, "Static MS"),
        (dur_sta_diff, "Static duration"),
    ]

    ims = []

    # Row 1: dynamic
    for i, (da, title) in enumerate(data_dyn, start=1):
        ax = make_clean_polar_ax(fig, nrows=2, ncols=3, index=i)
        im = plot_panel(ax, da, proj, VMAX_DIFF, title)
        ims.append(im)

    # Row 2: static
    for j, (da, title) in enumerate(data_sta, start=4):
        ax = make_clean_polar_ax(fig, nrows=2, ncols=3, index=j)
        im = plot_panel(ax, da, proj, VMAX_DIFF, title)
        ims.append(im)

    # Shared colorbar
    cax = fig.add_axes([0.15, 0.08, 0.7, 0.03])
    cb = fig.colorbar(ims[-1], cax=cax, orientation="horizontal")
    cb.set_label(
        f"Mean change (post − pre, days) "
        f"[pre={pre_s}–{pre_e}, post={post_s}–{post_e}]",
        fontsize=9,
    )
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_visible(False)

    fig.suptitle(
        f"FS, MS, and season duration anomalies (post − pre): "
        f"static vs dynamic\npre={pre_s}–{pre_e}, post={post_s}–{post_e}",
        fontsize=11,
        fontweight="bold",
        y=0.97,
    )

    fig.tight_layout(rect=[0.02, 0.14, 0.98, 0.95])

    # Save with period key in filename
    fig_name = (
        f"Fig_FS_MS_duration_anomaly_maps_static_vs_dynamic_{period_key}.png"
    )

    out_path = get_fig_path(
        project_root=PROJECT_ROOT,
        subfolder=SUBFOLDER,
        fig_name=fig_name,
    )

    save_and_upload(
        fig,
        out_path,
        remote_root=REMOTE_ROOT,
        remote_subdir=SUBFOLDER,
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    fields = load_fs_ms_clim_anom()

    for key, cfg in PERIOD_CONFIGS.items():
        print(f"\n[INFO] Making anomaly maps for period config: {key}")
        make_anomaly_map_for_period(key, cfg, fields)


if __name__ == "__main__":
    main()
