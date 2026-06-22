#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig04_crossing_frequency_maps.py

Climatological mean and linear trend in threshold-crossing frequency during
the seasonal window, for Freeze Start (FS) and Melt Start (MS), SMMR 1979-2023.

Panels:
  (a) FS mean crossing frequency (climatology)
  (b) MS mean crossing frequency (climatology)
  (c) FS linear trend in crossing frequency (crossings/year), stippled p<0.05
  (d) MS linear trend in crossing frequency, stippled p<0.05

Inputs:
  data/transition_metrics/SMMR/crossing_freq_FS_thr15.nc  (year, y, x)
  data/transition_metrics/SMMR/crossing_freq_MS_thr15.nc  (year, y, x)

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
from scipy import stats

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
METRICS_DIR = PROJECT_ROOT / "data" / "transition_metrics" / SENSOR
REMOTE_ROOT = "gdrive:sea-ice-phase/results/Ch2_Figures"
SUBFOLDER   = ""
VMAX_CLIM   = 6.0    # crossings — cap at p95
VMAX_TREND  = 0.06   # crossings/year
P_THRESH    = 0.05   # stippling threshold


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def load_crossing_freq(phase: str) -> xr.DataArray:
    path = METRICS_DIR / f"crossing_freq_{phase}_thr{THRESH_PCT}.nc"
    ds   = xr.open_dataset(path)
    cf   = ds["crossing_freq"].sel(year=slice(YEAR_MIN, YEAR_MAX))
    ds.close()
    return cf


def compute_trend(cf: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    """
    Pixel-wise linear trend in crossing frequency.
    Returns slope (crossings/year) and p-value arrays, both (y, x).
    """
    years  = cf.year.values.astype(float)
    values = cf.values   # (year, y, x)
    ny, nx = values.shape[1], values.shape[2]
    slope  = np.full((ny, nx), np.nan)
    pval   = np.full((ny, nx), np.nan)

    for j in range(ny):
        for i in range(nx):
            ts = values[:, j, i]
            if np.sum(np.isfinite(ts)) < 10:
                continue
            mask = np.isfinite(ts)
            s, _, _, p, _ = stats.linregress(years[mask], ts[mask])
            slope[j, i] = s
            pval[j, i]  = p

    return slope, pval


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


def add_stippling(ax, x, y, sig_mask, proj, density=3):
    """Stipple significant pixels."""
    xs = x[::density]
    ys = y[::density]
    sm = sig_mask[::density, ::density]
    xg, yg = np.meshgrid(xs, ys)
    ax.scatter(
        xg[sm], yg[sm],
        s=0.3, c="k", alpha=0.4,
        transform=proj, zorder=5,
        linewidths=0,
    )


# ---------------------------------------------------------------------
# MAIN PLOT
# ---------------------------------------------------------------------

def plot_crossing_freq_maps() -> None:
    set_mpl_defaults()

    print("Loading crossing frequency data...")
    cf_fs = load_crossing_freq("FS")
    cf_ms = load_crossing_freq("MS")

    # sentinel masks
    sentinel_fs = get_sentinel_mask(PROJECT_ROOT, "FS")
    sentinel_ms = get_sentinel_mask(PROJECT_ROOT, "MS")

    # climatological mean
    clim_fs = cf_fs.mean("year", skipna=True).values
    clim_ms = cf_ms.mean("year", skipna=True).values
    clim_fs[sentinel_fs.values if hasattr(sentinel_fs, 'values') else sentinel_fs] = np.nan
    clim_ms[sentinel_ms.values if hasattr(sentinel_ms, 'values') else sentinel_ms] = np.nan

    print("Computing trends (this may take a minute)...")
    slope_fs, pval_fs = compute_trend(cf_fs)
    slope_ms, pval_ms = compute_trend(cf_ms)
    slope_fs[sentinel_fs.values if hasattr(sentinel_fs, 'values') else sentinel_fs] = np.nan
    slope_ms[sentinel_ms.values if hasattr(sentinel_ms, 'values') else sentinel_ms] = np.nan

    sig_fs = (pval_fs < P_THRESH) & np.isfinite(slope_fs)
    sig_ms = (pval_ms < P_THRESH) & np.isfinite(slope_ms)

    x = cf_fs["x"].values
    y = cf_fs["y"].values
    proj = ccrs.SouthPolarStereo()

    cmap_clim  = plt.cm.YlOrRd
    norm_clim  = mcolors.Normalize(vmin=0, vmax=VMAX_CLIM)
    cmap_trend = plt.cm.RdBu_r
    norm_trend = mcolors.Normalize(vmin=-VMAX_TREND, vmax=VMAX_TREND)

    fig = plt.figure(figsize=(12.0, 10.0))

    panels = [
        ("(a) FS — climatological mean",      clim_fs,  221, cmap_clim,  norm_clim,  None,   "Mean crossings/season"),
        ("(b) MS — climatological mean",      clim_ms,  222, cmap_clim,  norm_clim,  None,   "Mean crossings/season"),
        ("(c) FS — trend (crossings yr⁻¹)",  slope_fs, 223, cmap_trend, norm_trend, sig_fs, "Trend (crossings yr⁻¹)"),
        ("(d) MS — trend (crossings yr⁻¹)",  slope_ms, 224, cmap_trend, norm_trend, sig_ms, "Trend (crossings yr⁻¹)"),
    ]

    # store last im per colormap for colorbars
    im_clim  = None
    im_trend = None

    for title, data, subplot_code, cmap, norm, sig, _ in panels:
        ax = make_polar_ax(fig, subplot_code)
        im = ax.pcolormesh(
            x, y, data,
            transform=proj,
            cmap=cmap,
            norm=norm,
            shading="auto",
            zorder=1,
        )
        if sig is not None:
            add_stippling(ax, x, y, sig, proj)
            im_trend = im
        else:
            im_clim = im
        ax.set_title(title, fontsize=9, pad=3)

    # colorbar for climatology panels (a, b)
    cax1 = fig.add_axes([0.08, 0.52, 0.38, 0.025])
    cb1  = fig.colorbar(im_clim, cax=cax1, orientation="horizontal", extend="max")
    cb1.set_label(f"Mean crossings per season ({YEAR_MIN}–{YEAR_MAX})", fontsize=8)
    cb1.ax.tick_params(labelsize=7)
    cb1.outline.set_visible(False)

    # colorbar for trend panels (c, d)
    cax2 = fig.add_axes([0.08, 0.06, 0.38, 0.025])
    cb2  = fig.colorbar(im_trend, cax=cax2, orientation="horizontal", extend="both")
    cb2.set_label("Linear trend (crossings yr⁻¹), stippled p<0.05", fontsize=8)
    cb2.ax.tick_params(labelsize=7)
    cb2.outline.set_visible(False)

    fig.suptitle(
        f"Threshold crossing frequency — {SENSOR} thr={THRESH_PCT}%",
        fontsize=11, y=0.98
    )
    fig.tight_layout(rect=[0, 0.10, 1, 0.97])

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
    plot_crossing_freq_maps()