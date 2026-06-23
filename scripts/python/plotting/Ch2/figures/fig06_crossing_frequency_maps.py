#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig06_crossing_frequency_maps.py

Pre/post-2016 change in threshold crossing frequency for FS and MS.

Panels:
  (a) FS — mean crossing frequency post-2016 minus pre-2016 (smoothed)
  (b) MS — mean crossing frequency post-2016 minus pre-2016 (smoothed)

Sector-mean statistics printed to stdout for caption/table use.

Inputs:
  data/transition_metrics/SMMR/crossing_freq_FS_thr15.nc  (year, y, x)
  data/transition_metrics/SMMR/crossing_freq_MS_thr15.nc  (year, y, x)
  data/canonical_sectors.nc

Output:
  results/Ch2_Figures/Fig04_crossing_frequency_FS_MS_SMMR_thr15_prepost2016.png
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
VMAX_DIFF   = 2.0
SMOOTH_SIZE = 5

SECTOR_NAMES = {
    1: "AB",
    2: "Weddell",
    3: "KH VII",
    4: "E. Antarctica",
    5: "Ross",
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
    nan_mask  = np.isnan(arr)
    filled    = np.where(nan_mask, 0.0, arr)
    weight    = np.where(nan_mask, 0.0, 1.0)
    s_filled  = uniform_filter(filled, size=size)
    s_weight  = uniform_filter(weight, size=size)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(s_weight > 0.1, s_filled / s_weight, np.nan)
    return out


def make_polar_ax(fig, row, col, nrows=1, ncols=2):
    proj = ccrs.SouthPolarStereo()
    ax   = fig.add_subplot(nrows, ncols, (row - 1) * ncols + col,
                           projection=proj)
    ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
    ax.add_feature(
        cfeature.LAND.with_scale("110m"),
        facecolor="0.85", edgecolor="0.6", linewidth=0.4, zorder=3,
    )
    ax.coastlines(linewidth=0.4, color="0.4", zorder=4)
    gl = ax.gridlines(draw_labels=False)
    gl.xlines = False
    gl.ylines = False
    ax.set_facecolor("white")
    return ax


def print_sector_stats(cf: xr.DataArray, phase: str,
                        sector_id: np.ndarray, sentinel: np.ndarray) -> None:
    pre  = cf.sel(year=slice(YEAR_MIN, PRE_END)).mean("year", skipna=True).values
    post = cf.sel(year=slice(POST_START, YEAR_MAX)).mean("year", skipna=True).values
    pre[sentinel]  = np.nan
    post[sentinel] = np.nan
    print(f"\n{phase} sector means (crossings/season):")
    print(f"  {'Sector':<16} {'Pre':>6} {'Post':>6} {'Δ':>6}")
    for sid, name in SECTOR_NAMES.items():
        mask = (sector_id == sid) & ~sentinel
        p1 = float(np.nanmean(pre[mask]))
        p2 = float(np.nanmean(post[mask]))
        print(f"  {name:<16} {p1:>6.3f} {p2:>6.3f} {p2-p1:>+6.3f}")


# ---------------------------------------------------------------------
# MAIN PLOT
# ---------------------------------------------------------------------

def plot_crossing_freq_diff() -> None:
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
    diff_fs_s = smooth(diff_fs, SMOOTH_SIZE)
    diff_ms_s = smooth(diff_ms, SMOOTH_SIZE)

    # print sector stats for caption
    print_sector_stats(cf_fs, "FS", sector_id, sent_fs)
    print_sector_stats(cf_ms, "MS", sector_id, sent_ms)

    x    = cf_fs["x"].values
    y    = cf_fs["y"].values
    proj = ccrs.SouthPolarStereo()

    cmap = plt.cm.RdBu_r
    norm = mcolors.Normalize(vmin=-VMAX_DIFF, vmax=VMAX_DIFF)

    fig = plt.figure(figsize=(10.0, 5.5))

    panels = [
        ("(a) FS — post minus pre 2016", diff_fs_s, 1),
        ("(b) MS — post minus pre 2016", diff_ms_s, 2),
    ]

    im_last = None
    for title, data, col in panels:
        ax = make_polar_ax(fig, 1, col)
        im_last = ax.pcolormesh(
            x, y, data,
            transform=proj,
            cmap=cmap,
            norm=norm,
            shading="auto",
            zorder=1,
        )
        ax.set_title(title, fontsize=10, pad=4)

    # shared colorbar
    cax = fig.add_axes([0.2, 0.06, 0.6, 0.03])
    cb  = fig.colorbar(im_last, cax=cax, orientation="horizontal", extend="both")
    cb.set_label(
        f"Δ mean crossings/season (post {POST_START}–{YEAR_MAX} minus pre {YEAR_MIN}–{PRE_END})",
        fontsize=9
    )
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_visible(False)

    fig.tight_layout(rect=[0, 0.12, 1, 1.0])

    fig_name = format_fig_name(
        num=4,
        short=f"crossing_frequency_FS_MS_{SENSOR}_thr{THRESH_PCT}_prepost2016",
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
    plot_crossing_freq_diff()