#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FS, MS, and season duration mean-change (post − pre): static vs dynamic.

For each period config (A, B, D):

Row 1: Dynamic  (FS, MS, duration)
Row 2: Static   (FS, MS, duration)

Each panel shows:
    mean(post) − mean(pre)   [days]

Inputs (precomputed):
  results/anomalies/FS_dynamic_climatology.nc   (FS_dynamic_clim[y,x])
  results/anomalies/FS_dynamic_anomalies.nc     (FS_dynamic_anom[year,y,x])
  results/anomalies/MS_dynamic_climatology.nc   (MS_dynamic_clim[y,x])
  results/anomalies/MS_dynamic_anomalies.nc     (MS_dynamic_anom[year,y,x])
  results/anomalies/FS_static_climatology.nc    (FS_static_clim[y,x])
  results/anomalies/FS_static_anomalies.nc      (FS_static_anom[year,y,x])
  results/anomalies/MS_static_climatology.nc    (MS_static_clim[y,x])
  results/anomalies/MS_static_anomalies.nc      (MS_static_anom[year,y,x])
"""

import sys
from pathlib import Path

import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ---------------------------------------------------------------------
# Shared utils
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
SECTOR_FILE = PROJECT_ROOT / "data" / "canonical_sectors.nc"

REMOTE_ROOT = "gdrive:sea-ice-phase/Results/Ch2_Figures"
SUBFOLDER   = "anomalies"

# ---------------------------------------------------------------------
# Period configs (same as before)
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

VMAX_DIFF = 30.0  # days, symmetric for post−pre maps


# ---------------------------------------------------------------------
# Load clim + anoms
# ---------------------------------------------------------------------
def load_fs_ms_clim_anom():
    fs_dyn_clim = xr.open_dataset(ANOM_DIR / "FS_dynamic_climatology.nc")["FS_dynamic_clim"]
    fs_dyn_anom = xr.open_dataset(ANOM_DIR / "FS_dynamic_anomalies.nc")["FS_dynamic_anom"]

    ms_dyn_clim = xr.open_dataset(ANOM_DIR / "MS_dynamic_climatology.nc")["MS_dynamic_clim"]
    ms_dyn_anom = xr.open_dataset(ANOM_DIR / "MS_dynamic_anomalies.nc")["MS_dynamic_anom"]

    fs_sta_clim = xr.open_dataset(ANOM_DIR / "FS_static_climatology.nc")["FS_static_clim"]
    fs_sta_anom = xr.open_dataset(ANOM_DIR / "FS_static_anomalies.nc")["FS_static_anom"]

    ms_sta_clim = xr.open_dataset(ANOM_DIR / "MS_static_climatology.nc")["MS_static_clim"]
    ms_sta_anom = xr.open_dataset(ANOM_DIR / "MS_static_anomalies.nc")["MS_static_anom"]

    try:
        ds_mask = xr.open_dataset(SECTOR_FILE)
        valid_ocean = ds_mask["valid_ocean"].astype(bool)
        ds_mask.close()
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
# Compute post−pre means
# ---------------------------------------------------------------------
def compute_pre_post(clim, anom, pre_start, pre_end, post_start, post_end):
    """
    Given climatology (clim[y,x]) and anomalies anom[year,y,x] defined as:
      anom(year) = field(year) − clim

    Reconstruct mean fields over pre and post periods:

      pre_mean  = mean(field(year)   for year in [pre_start, pre_end])
      post_mean = mean(field(year)   for year in [post_start, post_end])

    and return (pre_mean, post_mean, post_minus_pre).
    """
    anom_pre = anom.sel(year=slice(pre_start, pre_end))
    anom_post = anom.sel(year=slice(post_start, post_end))

    pre_mean = clim + anom_pre.mean("year", skipna=True)
    post_mean = clim + anom_post.mean("year", skipna=True)
    diff = post_mean - pre_mean
    return pre_mean, post_mean, diff


# ---------------------------------------------------------------------
# Map helpers
# ---------------------------------------------------------------------
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
        vmax=+vlim,
        shading="auto",
    )
    ax.set_title(title, fontsize=9, fontweight="bold")
    return im


# ---------------------------------------------------------------------
# Make one 2×3 figure for a given period config
# ---------------------------------------------------------------------
def make_anomaly_map_for_period(period_key, cfg, fields):
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

    # Dynamic pre/post + duration
    _, _, fs_dyn_diff = compute_pre_post(fs_dyn_clim, fs_dyn_anom, pre_s, pre_e, post_s, post_e)
    _, _, ms_dyn_diff = compute_pre_post(ms_dyn_clim, ms_dyn_anom, pre_s, pre_e, post_s, post_e)
    dur_dyn_pre, dur_dyn_post, _ = compute_pre_post(
        fs_dyn_clim - ms_dyn_clim,  # not actually used
        fs_dyn_anom * 0,            # dummy
        pre_s, pre_e, post_s, post_e,
    )
    # Better: compute durations directly from FS/MS means:
    fs_dyn_pre, fs_dyn_post, _ = compute_pre_post(fs_dyn_clim, fs_dyn_anom, pre_s, pre_e, post_s, post_e)
    ms_dyn_pre, ms_dyn_post, _ = compute_pre_post(ms_dyn_clim, ms_dyn_anom, pre_s, pre_e, post_s, post_e)
    dur_dyn_diff = (ms_dyn_post - fs_dyn_post) - (ms_dyn_pre - fs_dyn_pre)

    # Static pre/post + duration
    _, _, fs_sta_diff = compute_pre_post(fs_sta_clim, fs_sta_anom, pre_s, pre_e, post_s, post_e)
    _, _, ms_sta_diff = compute_pre_post(ms_sta_clim, ms_sta_anom, pre_s, pre_e, post_s, post_e)
    fs_sta_pre, fs_sta_post, _ = compute_pre_post(fs_sta_clim, fs_sta_anom, pre_s, pre_e, post_s, post_e)
    ms_sta_pre, ms_sta_post, _ = compute_pre_post(ms_sta_clim, ms_sta_anom, pre_s, pre_e, post_s, post_e)
    dur_sta_diff = (ms_sta_post - fs_sta_post) - (ms_sta_pre - fs_sta_pre)

    if valid_ocean is not None:
        fs_dyn_diff = fs_dyn_diff.where(valid_ocean)
        ms_dyn_diff = ms_dyn_diff.where(valid_ocean)
        dur_dyn_diff = dur_dyn_diff.where(valid_ocean)

        fs_sta_diff = fs_sta_diff.where(valid_ocean)
        ms_sta_diff = ms_sta_diff.where(valid_ocean)
        dur_sta_diff = dur_sta_diff.where(valid_ocean)

    fig = plt.figure(figsize=(9, 6))
    ims = []

    # Row 1: dynamic
    ax, proj = make_clean_polar_ax(fig, 2, 3, 1)
    ims.append(plot_panel(ax, fs_dyn_diff, proj, VMAX_DIFF, "Dynamic FS"))

    ax, proj = make_clean_polar_ax(fig, 2, 3, 2)
    ims.append(plot_panel(ax, ms_dyn_diff, proj, VMAX_DIFF, "Dynamic MS"))

    ax, proj = make_clean_polar_ax(fig, 2, 3, 3)
    ims.append(plot_panel(ax, dur_dyn_diff, proj, VMAX_DIFF, "Dynamic duration"))

    # Row 2: static
    ax, proj = make_clean_polar_ax(fig, 2, 3, 4)
    ims.append(plot_panel(ax, fs_sta_diff, proj, VMAX_DIFF, "Static FS"))

    ax, proj = make_clean_polar_ax(fig, 2, 3, 5)
    ims.append(plot_panel(ax, ms_sta_diff, proj, VMAX_DIFF, "Static MS"))

    ax, proj = make_clean_polar_ax(fig, 2, 3, 6)
    ims.append(plot_panel(ax, dur_sta_diff, proj, VMAX_DIFF, "Static duration"))

    # shared colorbar
    cax = fig.add_axes([0.15, 0.08, 0.7, 0.03])
    cb = fig.colorbar(ims[-1], cax=cax, orientation="horizontal")
    cb.set_label(
        f"Mean change (post − pre, days) [pre={pre_s}–{pre_e}, post={post_s}–{post_e}]",
        fontsize=9,
    )
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_visible(False)

    fig.tight_layout(rect=[0.02, 0.14, 0.98, 0.95])

    fig_name = f"Fig_FS_MS_duration_anomaly_maps_static_vs_dynamic_{period_key}.png"
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


def main():
    fields = load_fs_ms_clim_anom()
    for key, cfg in PERIOD_CONFIGS.items():
        print(f"[INFO] Making maps for period config: {key}")
        make_anomaly_map_for_period(key, cfg, fields)


if __name__ == "__main__":
    main()
