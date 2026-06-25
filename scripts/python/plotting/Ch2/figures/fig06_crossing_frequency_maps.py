#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig06_crossing_frequency_maps.py

Pre/post-2016 change in threshold crossing frequency for FS and MS.

Panels:
  (a) FS — mean crossing frequency post-2016 minus pre-2016 (smoothed)
  (b) MS — mean crossing frequency post-2016 minus pre-2016 (smoothed)

Sector-mean statistics printed to stdout for caption/table use.
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

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.python.plotting.ch2_fig_utils import (
    set_mpl_defaults, format_fig_name, get_fig_path, save_and_upload, get_sentinel_mask,
)

SENSOR      = "SMMR"
THRESH_PCT  = 15
YEAR_MIN    = 1979
YEAR_MAX    = 2024
PRE_END     = 2015
POST_START  = 2016
METRICS_DIR = PROJECT_ROOT / "data" / "transition_metrics" / SENSOR
SECTOR_FILE = PROJECT_ROOT / "data" / "canonical_sectors.nc"
REMOTE_ROOT = "gdrive:sea-ice-phase/results/Ch2_Figures"
SUBFOLDER   = ""
VMAX_DIFF   = 2.0
SMOOTH_SIZE = 5
SECTOR_NAMES = {1:"AB", 2:"Weddell", 3:"KH VII", 4:"E. Antarctica", 5:"Ross"}

def load_crossing_freq(phase):
    path = METRICS_DIR / f"crossing_freq_{phase}_thr{THRESH_PCT}.nc"
    ds = xr.open_dataset(path)
    cf = ds["crossing_freq"].sel(year=slice(YEAR_MIN, YEAR_MAX))
    ds.close()
    return cf

def smooth(arr, size):
    nan_mask = np.isnan(arr)
    filled   = np.where(nan_mask, 0.0, arr)
    weight   = np.where(nan_mask, 0.0, 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(uniform_filter(weight, size=size) > 0.1,
                       uniform_filter(filled, size=size) / uniform_filter(weight, size=size),
                       np.nan)
    return out

def make_polar_ax(fig, row, col, nrows=2, ncols=3):
    proj = ccrs.SouthPolarStereo()
    ax   = fig.add_subplot(nrows, ncols, (row-1)*ncols + col, projection=proj)
    ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND.with_scale("110m"),
                   facecolor="0.85", edgecolor="0.6", linewidth=0.4, zorder=3)
    ax.coastlines(linewidth=0.4, color="0.4", zorder=4)
    gl = ax.gridlines(draw_labels=False)
    gl.xlines = False
    gl.ylines = False
    ax.set_facecolor("white")
    return ax

def print_sector_stats(cf, phase, sector_id, sentinel):
    pre  = cf.sel(year=slice(YEAR_MIN, PRE_END)).mean("year", skipna=True).values
    post = cf.sel(year=slice(POST_START, YEAR_MAX)).mean("year", skipna=True).values
    pre[sentinel] = np.nan
    post[sentinel] = np.nan
    print(f"\n{phase} sector means (crossings/season):")
    print(f"  {'Sector':<16} {'Pre':>6} {'Post':>6} {'Delta':>7}")
    for sid, name in SECTOR_NAMES.items():
        mask = (sector_id == sid) & ~sentinel
        p1 = float(np.nanmean(pre[mask]))
        p2 = float(np.nanmean(post[mask]))
        print(f"  {name:<16} {p1:>6.3f} {p2:>6.3f} {p2-p1:>+7.3f}")

def plot_crossing_freq_diff():
    set_mpl_defaults()
    print("Loading data...")
    cf_fs = load_crossing_freq("FS")
    cf_ms = load_crossing_freq("MS")

    ds_sec    = xr.open_dataset(SECTOR_FILE)
    sector_id = ds_sec["sector_id"].values.astype(int)
    ds_sec.close()

    sent_fs = get_sentinel_mask(PROJECT_ROOT, "FS")
    sent_ms = get_sentinel_mask(PROJECT_ROOT, "MS")
    sent_fs = sent_fs.values if hasattr(sent_fs, "values") else sent_fs
    sent_ms = sent_ms.values if hasattr(sent_ms, "values") else sent_ms

    diff_fs = (cf_fs.sel(year=slice(POST_START, YEAR_MAX)).mean("year", skipna=True).values -
               cf_fs.sel(year=slice(YEAR_MIN, PRE_END)).mean("year", skipna=True).values)
    diff_ms = (cf_ms.sel(year=slice(POST_START, YEAR_MAX)).mean("year", skipna=True).values -
               cf_ms.sel(year=slice(YEAR_MIN, PRE_END)).mean("year", skipna=True).values)
    diff_fs[sent_fs] = np.nan
    diff_ms[sent_ms] = np.nan

    print("Smoothing...")
    diff_fs_s = smooth(diff_fs, SMOOTH_SIZE)
    diff_ms_s = smooth(diff_ms, SMOOTH_SIZE)

    print_sector_stats(cf_fs, "FS", sector_id, sent_fs)
    print_sector_stats(cf_ms, "MS", sector_id, sent_ms)

    x    = cf_fs["x"].values
    y    = cf_fs["y"].values
    proj = ccrs.SouthPolarStereo()
    cmap = plt.cm.RdBu_r
    norm = mcolors.Normalize(vmin=-VMAX_DIFF, vmax=VMAX_DIFF)

    # pre and post means (unsmoothed for absolute maps, smoothed for diff)
    fs_pre_m  = smooth(cf_fs.sel(year=slice(YEAR_MIN, PRE_END)).mean("year", skipna=True).values, SMOOTH_SIZE)
    fs_post_m = smooth(cf_fs.sel(year=slice(POST_START, YEAR_MAX)).mean("year", skipna=True).values, SMOOTH_SIZE)
    ms_pre_m  = smooth(cf_ms.sel(year=slice(YEAR_MIN, PRE_END)).mean("year", skipna=True).values, SMOOTH_SIZE)
    ms_post_m = smooth(cf_ms.sel(year=slice(POST_START, YEAR_MAX)).mean("year", skipna=True).values, SMOOTH_SIZE)
    fs_pre_m[sent_fs]  = np.nan
    fs_post_m[sent_fs] = np.nan
    ms_pre_m[sent_ms]  = np.nan
    ms_post_m[sent_ms] = np.nan

    cmap_abs  = plt.cm.YlOrRd
    norm_abs  = mcolors.Normalize(vmin=0, vmax=5)

    fig = plt.figure(figsize=(14.0, 9.0))
    panels = [
        (1, 1, "(a) FS pre-2016 (1979–2015)",   fs_pre_m,  cmap_abs, norm_abs),
        (1, 2, "(b) FS post-2016 (2016–2024)",  fs_post_m, cmap_abs, norm_abs),
        (1, 3, "(c) FS difference (post−pre)",   diff_fs_s, cmap,     norm),
        (2, 1, "(d) MS pre-2016 (1979–2015)",   ms_pre_m,  cmap_abs, norm_abs),
        (2, 2, "(e) MS post-2016 (2016–2024)",  ms_post_m, cmap_abs, norm_abs),
        (2, 3, "(f) MS difference (post−pre)",   diff_ms_s, cmap,     norm),
    ]

    im_abs = None
    im_diff = None
    for row, col, title, data, cm, nm in panels:
        ax = make_polar_ax(fig, row, col)
        im = ax.pcolormesh(x, y, data, transform=proj,
                           cmap=cm, norm=nm, shading="auto", zorder=1)
        ax.set_title(title, fontsize=9, pad=4, fontweight="bold")
        if col in (1, 2):
            im_abs = im
        else:
            im_diff = im

    # colorbar for absolute maps
    cax1 = fig.add_axes([0.05, 0.04, 0.55, 0.02])
    cb1  = fig.colorbar(im_abs, cax=cax1, orientation="horizontal")
    cb1.set_label("Mean crossings/season", fontsize=9)
    cb1.ax.tick_params(labelsize=8)
    cb1.outline.set_visible(False)

    # colorbar for difference maps
    cax2 = fig.add_axes([0.65, 0.04, 0.30, 0.02])
    cb2  = fig.colorbar(im_diff, cax=cax2, orientation="horizontal", extend="both")
    cb2.set_label(f"Δ crossings/season", fontsize=9)
    cb2.ax.tick_params(labelsize=8)
    cb2.outline.set_visible(False)

    fig.tight_layout(rect=[0, 0.08, 1, 1.0])

    fig_name = format_fig_name(
        num=7,
        short=f"crossing_frequency_FS_MS_{SENSOR}_thr{THRESH_PCT}_prepost2016",
    )
    out_path = get_fig_path(project_root=PROJECT_ROOT, subfolder=SUBFOLDER, fig_name=fig_name)
    save_and_upload(fig, out_path, remote_root=REMOTE_ROOT, remote_subdir=SUBFOLDER)

if __name__ == "__main__":
    plot_crossing_freq_diff()
