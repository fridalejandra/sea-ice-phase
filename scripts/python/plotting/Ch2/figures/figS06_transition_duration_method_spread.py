#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
figS06_transition_duration_method_spread.py

Supplementary figure showing spatial patterns of transition ambiguity
and method sensitivity for SMMR 1979-2023.

Panels:
  (a) FS — fraction of years with non-zero transition duration (thr=15%)
  (b) MS — fraction of years with non-zero transition duration (thr=15%)
  (c) FS — climatological mean method spread across static thr/k combinations
  (d) MS — climatological mean method spread across static thr/k combinations

Inputs:
  data/transition_metrics/SMMR/transition_dur_FS_thr15.nc  (year, y, x)
  data/transition_metrics/SMMR/transition_dur_MS_thr15.nc  (year, y, x)
  data/transition_metrics/SMMR/method_spread_FS.nc         (year, y, x)
  data/transition_metrics/SMMR/method_spread_MS.nc         (year, y, x)

Output:
  results/Ch2_Figures/FigS06_transition_duration_method_spread_SMMR.png
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
# project root
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
YEAR_MAX    = 2024
METRICS_DIR = PROJECT_ROOT / "data" / "transition_metrics" / SENSOR
REMOTE_ROOT = "gdrive:sea-ice-phase/results/Ch2_Figures"
SUBFOLDER   = ""
SMOOTH_SIZE = 9
VMAX_SPREAD = 15.0   # days — method spread colorscale cap


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def load(fname: str, varname: str, year_min: int, year_max: int) -> xr.DataArray:
    ds = xr.open_dataset(METRICS_DIR / fname)
    da = ds[varname].sel(year=slice(year_min, year_max))
    ds.close()
    return da


def smooth(arr: np.ndarray, size: int) -> np.ndarray:
    nan_mask = np.isnan(arr)
    filled   = np.where(nan_mask, 0.0, arr)
    weight   = np.where(nan_mask, 0.0, 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(
            uniform_filter(weight, size=size) > 0.1,
            uniform_filter(filled, size=size) / uniform_filter(weight, size=size),
            np.nan,
        )
    return out


def frac_nonzero(da: xr.DataArray, sentinel: np.ndarray) -> np.ndarray:
    """
    Fraction of years where transition_dur > 0, per pixel.
    NaN where fewer than 5 valid years.
    """
    vals       = da.values   # (year, y, x)
    valid      = np.isfinite(vals)
    n_valid    = valid.sum(axis=0)
    n_nonzero  = ((vals > 0) & valid).sum(axis=0)
    with np.errstate(invalid="ignore"):
        frac = np.where(n_valid >= 5, n_nonzero / n_valid, np.nan)
    frac[sentinel] = np.nan
    return frac


def clim_mean(da: xr.DataArray, sentinel: np.ndarray) -> np.ndarray:
    """Climatological mean, NaN-masked."""
    arr = da.mean("year", skipna=True).values
    arr[sentinel] = np.nan
    return arr


def make_polar_ax(fig, nrows, ncols, pos):
    proj = ccrs.SouthPolarStereo()
    ax   = fig.add_subplot(nrows, ncols, pos, projection=proj)
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


# ---------------------------------------------------------------------
# MAIN PLOT
# ---------------------------------------------------------------------

def plot_figS06() -> None:
    set_mpl_defaults()

    print("Loading data...")
    dur_fs = load(f"transition_dur_FS_thr{THRESH_PCT}.nc", "transition_dur", YEAR_MIN, YEAR_MAX)
    dur_ms = load(f"transition_dur_MS_thr{THRESH_PCT}.nc", "transition_dur", YEAR_MIN, YEAR_MAX)
    spr_fs = load("method_spread_FS.nc", "method_spread", YEAR_MIN, YEAR_MAX)
    spr_ms = load("method_spread_MS.nc", "method_spread", YEAR_MIN, YEAR_MAX)

    sent_fs = get_sentinel_mask(PROJECT_ROOT, "FS")
    sent_ms = get_sentinel_mask(PROJECT_ROOT, "MS")
    sent_fs = sent_fs.values if hasattr(sent_fs, "values") else sent_fs
    sent_ms = sent_ms.values if hasattr(sent_ms, "values") else sent_ms

    print("Computing fraction of ambiguous years...")
    frac_fs = smooth(frac_nonzero(dur_fs, sent_fs), SMOOTH_SIZE)
    frac_ms = smooth(frac_nonzero(dur_ms, sent_ms), SMOOTH_SIZE)

    print("Computing method spread climatology...")
    mean_spr_fs = smooth(clim_mean(spr_fs, sent_fs), SMOOTH_SIZE)
    mean_spr_ms = smooth(clim_mean(spr_ms, sent_ms), SMOOTH_SIZE)

    x = dur_fs["x"].values
    y = dur_fs["y"].values
    proj = ccrs.SouthPolarStereo()

    cmap_frac  = plt.cm.YlOrRd
    norm_frac  = mcolors.Normalize(vmin=0, vmax=1)
    cmap_spr   = plt.cm.plasma
    norm_spr   = mcolors.Normalize(vmin=0, vmax=VMAX_SPREAD)

    fig = plt.figure(figsize=(12.0, 12.0))

    panels = [
        ("(a) FS — fraction of years\nwith ambiguous transition", frac_fs,    1, cmap_frac, norm_frac),
        ("(b) MS — fraction of years\nwith ambiguous transition", frac_ms,    2, cmap_frac, norm_frac),
        ("(c) FS — mean method spread\nacross static thr/k",      mean_spr_fs, 3, cmap_spr,  norm_spr),
        ("(d) MS — mean method spread\nacross static thr/k",      mean_spr_ms, 4, cmap_spr,  norm_spr),
    ]

    im_frac = None
    im_spr  = None

    for title, data, pos, cmap, norm in panels:
        ax = make_polar_ax(fig, 2, 2, pos)
        im = ax.pcolormesh(
            x, y, data,
            transform=proj,
            cmap=cmap,
            norm=norm,
            shading="auto",
            zorder=1,
        )
        ax.set_title(title, fontsize=9, pad=3)
        if cmap == cmap_frac:
            im_frac = im
        else:
            im_spr = im

    # colorbar for fraction panels
    cax1 = fig.add_axes([0.08, 0.50, 0.38, 0.013])
    cb1  = fig.colorbar(im_frac, cax=cax1, orientation="horizontal")
    cb1.set_label(f"Fraction of years with transition duration > 0 ({YEAR_MIN}–{YEAR_MAX})", fontsize=8)
    cb1.ax.tick_params(labelsize=7)
    cb1.outline.set_visible(False)

    # colorbar for spread panels
    cax2 = fig.add_axes([0.08, 0.04, 0.38, 0.013])
    cb2  = fig.colorbar(im_spr, cax=cax2, orientation="horizontal", extend="max")
    cb2.set_label("Mean method spread across static thr/k combinations (days)", fontsize=8)
    cb2.ax.tick_params(labelsize=7)
    cb2.outline.set_visible(False)

    fig.suptitle(
        f"Transition ambiguity and method sensitivity — {SENSOR} {YEAR_MIN}–{YEAR_MAX}",
        fontsize=11, y=0.99,
    )
    fig.subplots_adjust(left=0.05, right=0.95, top=0.94, bottom=0.10, hspace=0.35, wspace=0.05)

    fig_name = f"FigS05_transition_duration_method_spread_{SENSOR}.png"

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
    plot_figS06()