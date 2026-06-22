#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig04_crossing_frequency_maps.py

2x3 figure showing threshold crossing frequency for FS and MS.

Panels:
  (a) FS climatological mean crossing frequency (smoothed)
  (b) MS climatological mean crossing frequency (smoothed)
  (c) FS pre/post-2016 difference (smoothed)
  (d) MS pre/post-2016 difference (smoothed)
  (e) FS sector bar chart — mean crossing freq pre vs post 2016
  (f) MS sector bar chart — mean crossing freq pre vs post 2016

Inputs:
  data/transition_metrics/SMMR/crossing_freq_FS_thr15.nc  (year, y, x)
  data/transition_metrics/SMMR/crossing_freq_MS_thr15.nc  (year, y, x)
  data/canonical_sectors.nc

Output:
  results/Ch2_Figures/Fig04_crossing_frequency_FS_MS_SMMR_thr15.png
"""

import sys
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.ndimage import uniform_filter

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
    get_sentinel_mask,
)

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
SENSOR      = "SMMR"
THRESH_PCT  = 15
YEAR_MIN    = 1979
YEAR_MAX    = 2023
PRE_END     = 2015
POST_START  = 2016
METRICS_DIR = PROJECT_ROOT / "data" / "transition_metrics" / SENSOR
SECTOR_FILE = PROJECT_ROOT / "data" / "canonical_sectors.nc"
REMOTE_ROOT = "gdrive:sea-ice-phase/results/Ch2_Figures"
SUBFOLDER   = ""
VMAX_CLIM   = 6.0    # crossings — cap at p95
VMAX_DIFF   = 2.0    # crossings difference
SMOOTH_SIZE = 5      # spatial smoothing kernel (pixels)

SECTOR_NAMES = {
    1: "AB",
    2: "Weddell",
    3: "KH VII",
    4: "E. Antarctica",
    5: "Ross",
}
SECTOR_COLORS = {
    "pre":  "#4A90C4",
    "post": "#C0392B",
}


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def load_crossing_freq(phase: str) -> xr.DataArray:
    path = METRICS_DIR / f"crossing_freq_{phase}_thr{THRESH_PCT}.nc"
    ds   = xr.open_dataset(path)
    cf   = ds["crossing_freq"].sel(year=slice(YEAR_MIN, YEAR_MAX))
    ds.close()
    return cf


def smooth(arr: np.ndarray, size: int) -> np.ndarray:
    """Spatial smoothing with NaN handling."""
    out = arr.copy()
    nan_mask = np.isnan(arr)
    arr_filled = np.where(nan_mask, 0.0, arr)
    weight = np.where(nan_mask, 0.0, 1.0)
    smoothed = uniform_filter(arr_filled, size=size)
    weight_s = uniform_filter(weight, size=size)
    with np.errstate(invalid="ignore"):
        out = np.where(weight_s > 0.1, smoothed / weight_s, np.nan)
    out[nan_mask & (weight_s < 0.1)] = np.nan
    return out


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


def sector_means(cf: xr.DataArray, sector_id: np.ndarray,
                 sentinel: np.ndarray) -> dict:
    """
    Compute pre/post mean crossing frequency per sector.
    Returns {sector_id: (pre_mean, post_mean)}.
    """
    pre  = cf.sel(year=slice(YEAR_MIN, PRE_END)).mean("year", skipna=True).values
    post = cf.sel(year=slice(POST_START, YEAR_MAX)).mean("year", skipna=True).values
    pre[sentinel]  = np.nan
    post[sentinel] = np.nan

    result = {}
    for sid in range(1, 6):
        mask = (sector_id == sid) & ~sentinel
        result[sid] = (
            float(np.nanmean(pre[mask])),
            float(np.nanmean(post[mask])),
        )
    return result


# ---------------------------------------------------------------------
# MAIN PLOT
# ---------------------------------------------------------------------

def plot_crossing_freq() -> None:
    set_mpl_defaults()

    print("Loading data...")
    cf_fs = load_crossing_freq("FS")
    cf_ms = load_crossing_freq("MS")

    ds_sec    = xr.open_dataset(SECTOR_FILE)
    sector_id = ds_sec["sector_id"].values.astype(int)
    ds_sec.close()

    sentinel_fs = get_sentinel_mask(PROJECT_ROOT, "FS")
    sentinel_ms = get_sentinel_mask(PROJECT_ROOT, "MS")
    sent_fs = sentinel_fs.values if hasattr(sentinel_fs, "values") else sentinel_fs
    sent_ms = sentinel_ms.values if hasattr(sentinel_ms, "values") else sentinel_ms

    # climatological means
    clim_fs = cf_fs.mean("year", skipna=True).values
    clim_ms = cf_ms.mean("year", skipna=True).values
    clim_fs[sent_fs] = np.nan
    clim_ms[sent_ms] = np.nan

    # pre/post difference
    pre_fs  = cf_fs.sel(year=slice(YEAR_MIN, PRE_END)).mean("year", skipna=True).values
    post_fs = cf_fs.sel(year=slice(POST_START, YEAR_MAX)).mean("year", skipna=True).values
    pre_ms  = cf_ms.sel(year=slice(YEAR_MIN, PRE_END)).mean("year", skipna=True).values
    post_ms = cf_ms.sel(year=slice(POST_START, YEAR_MAX)).mean("year", skipna=True).values

    diff_fs = post_fs - pre_fs
    diff_ms = post_ms - pre_ms
    diff_fs[sent_fs] = np.nan
    diff_ms[sent_ms] = np.nan

    # smooth
    print("Smoothing...")
    clim_fs_s = smooth(clim_fs, SMOOTH_SIZE)
    clim_ms_s = smooth(clim_ms, SMOOTH_SIZE)
    diff_fs_s = smooth(diff_fs, SMOOTH_SIZE)
    diff_ms_s = smooth(diff_ms, SMOOTH_SIZE)

    # sector means
    sec_fs = sector_means(cf_fs, sector_id, sent_fs)
    sec_ms = sector_means(cf_ms, sector_id, sent_ms)

    x = cf_fs["x"].values
    y = cf_fs["y"].values
    proj = ccrs.SouthPolarStereo()

    cmap_clim  = plt.cm.YlOrRd
    norm_clim  = mcolors.Normalize(vmin=0, vmax=VMAX_CLIM)
    cmap_diff  = plt.cm.RdBu_r
    norm_diff  = mcolors.Normalize(vmin=-VMAX_DIFF, vmax=VMAX_DIFF)

    fig = plt.figure(figsize=(14.0, 10.0))

    # --- map panels ---
    map_panels = [
        ("(a) FS — climatological mean", clim_fs_s, (3, 4, 1),  cmap_clim,  norm_clim),
        ("(b) MS — climatological mean", clim_ms_s, (3, 4, 2),  cmap_clim,  norm_clim),
        ("(c) FS — post minus pre 2016", diff_fs_s, (3, 4, 5),  cmap_diff,  norm_diff),
        ("(d) MS — post minus pre 2016", diff_ms_s, (3, 4, 6),  cmap_diff,  norm_diff),
    ]

    im_clim = None
    im_diff = None

    for title, data, pos, cmap, norm in map_panels:
        ax = make_polar_ax(fig, pos)
        im = ax.pcolormesh(
            x, y, data,
            transform=proj,
            cmap=cmap,
            norm=norm,
            shading="auto",
            zorder=1,
        )
        ax.set_title(title, fontsize=9, pad=3)
        if cmap == cmap_clim:
            im_clim = im
        else:
            im_diff = im

    # colorbars for map rows
    cax1 = fig.add_axes([0.13, 0.645, 0.30, 0.018])
    cb1  = fig.colorbar(im_clim, cax=cax1, orientation="horizontal", extend="max")
    cb1.set_label(f"Mean crossings/season ({YEAR_MIN}–{YEAR_MAX})", fontsize=7)
    cb1.ax.tick_params(labelsize=6)
    cb1.outline.set_visible(False)

    cax2 = fig.add_axes([0.13, 0.355, 0.30, 0.018])
    cb2  = fig.colorbar(im_diff, cax=cax2, orientation="horizontal", extend="both")
    cb2.set_label(f"Δ crossings/season (post {POST_START} − pre)", fontsize=7)
    cb2.ax.tick_params(labelsize=6)
    cb2.outline.set_visible(False)

    # --- bar chart panels (e) and (f) ---
    sector_ids   = list(range(1, 6))
    sector_labels = [SECTOR_NAMES[s] for s in sector_ids]
    x_pos        = np.arange(len(sector_ids))
    width        = 0.35

    for idx, (phase, sec_data, subplot_pos, panel_label) in enumerate([
        ("FS", sec_fs, (3, 4, 9),  "(e) FS — sector means"),
        ("MS", sec_ms, (3, 4, 10), "(f) MS — sector means"),
    ]):
        ax = fig.add_subplot(subplot_pos)
        pre_vals  = [sec_data[s][0] for s in sector_ids]
        post_vals = [sec_data[s][1] for s in sector_ids]

        ax.bar(x_pos - width/2, pre_vals,  width, label=f"{YEAR_MIN}–{PRE_END}",
               color=SECTOR_COLORS["pre"],  alpha=0.85, edgecolor="none")
        ax.bar(x_pos + width/2, post_vals, width, label=f"{POST_START}–{YEAR_MAX}",
               color=SECTOR_COLORS["post"], alpha=0.85, edgecolor="none")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(sector_labels, fontsize=7, rotation=20, ha="right")
        ax.set_ylabel("Mean crossings/season", fontsize=7)
        ax.set_title(panel_label, fontsize=9, pad=3)
        ax.tick_params(axis="y", labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        if idx == 0:
            ax.legend(fontsize=7, frameon=False)

    fig.suptitle(
        f"Threshold crossing frequency — {SENSOR} thr={THRESH_PCT}%",
        fontsize=11, y=0.99
    )

    fig_name = format_fig_name(
        num=4,
        short=f"crossing_frequency_FS_MS_{SENSOR}_thr{THRESH_PCT}",
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
    plot_crossing_freq()