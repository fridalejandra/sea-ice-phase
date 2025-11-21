#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fig_FS_MS_anomaly_timeseries_dynamic.py

Handcock-style anomaly time series with BOTH dynamic and static:

  (a) Season duration anomaly (MS − FS)
  (b) MS anomaly (retreat)
  (c) FS anomaly (advance)

Inputs (all already computed elsewhere):

  results/anomalies/FS_dynamic_anomalies.nc  -> FS_dynamic_anom(year,y,x)
  results/anomalies/MS_dynamic_anomalies.nc  -> MS_dynamic_anom(year,y,x)
  results/anomalies/FS_static_anomalies.nc   -> FS_static_anom(year,y,x)
  results/anomalies/MS_static_anomalies.nc   -> MS_static_anom(year,y,x)

Features:
  - area-weighted mean anomalies over valid ocean
  - DYNAMIC: raw yearly curves (thin grey), LOESS smooth (black), ±1σ band
  - STATIC: LOESS smooth only (blue)
  - highlight 2016, 2022, 2023 (red dots)
"""

import sys
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import statsmodels.api as sm

# ---------------------------------------------------------------------
# Ensure ch2_fig_utils is importable
# ---------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
PLOTTING_ROOT = HERE.parent  # .../plotting

if str(PLOTTING_ROOT) not in sys.path:
    sys.path.insert(0, str(PLOTTING_ROOT))

from ch2_fig_utils import (
    set_mpl_defaults,
    get_fig_path,
    save_and_upload,
    PROJECT_ROOT_CLUSTER,
)

set_mpl_defaults()

# project paths
PROJECT_ROOT = PROJECT_ROOT_CLUSTER
ANOM_DIR = PROJECT_ROOT / "results" / "anomalies"
SECTOR_FILE = PROJECT_ROOT / "data" / "canonical_sectors.nc"

# highlight years
YEARS_HI = [2016, 2022, 2023]


# ---------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------
def load_dynamic_anoms():
    fs = xr.open_dataset(ANOM_DIR / "FS_dynamic_anomalies.nc")["FS_dynamic_anom"]
    ms = xr.open_dataset(ANOM_DIR / "MS_dynamic_anomalies.nc")["MS_dynamic_anom"]
    return fs, ms


def load_static_anoms():
    fs = xr.open_dataset(ANOM_DIR / "FS_static_anomalies.nc")["FS_static_anom"]
    ms = xr.open_dataset(ANOM_DIR / "MS_static_anomalies.nc")["MS_static_anom"]
    return fs, ms


def load_ocean_mask():
    ds = xr.open_dataset(SECTOR_FILE)
    return ds["valid_ocean"].astype(bool).values


# ---------------------------------------------------------------------
# AREA-WEIGHTED MEAN
# ---------------------------------------------------------------------
def compute_area_weighted_mean(da, ocean_mask):
    """
    da: (year, y, x)
    ocean_mask: (y, x)
    """
    # If lat is available, use cosine-lat weights. Otherwise uniform.
    if "lat" in da.coords:
        lat = da["lat"].values
        w1d = np.cos(np.deg2rad(lat))  # [y]
        W = np.repeat(w1d[:, None], da.shape[2], axis=1)  # [y,x]
    else:
        W = np.ones_like(ocean_mask, dtype=float)

    W = W * ocean_mask
    W = W / np.nansum(W)

    years = da["year"].values
    out = []
    for y in years:
        arr = da.sel(year=y).values
        out.append(np.nansum(arr * W))

    return years, np.array(out)


# ---------------------------------------------------------------------
# LOESS SMOOTHING
# ---------------------------------------------------------------------
def loess_smooth(x, y, frac=0.25):
    lo = sm.nonparametric.lowess(y, x, frac=frac, return_sorted=True)
    return lo[:, 0], lo[:, 1]


# ---------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------
def plot_anomaly_timeseries():

    # load anomalies
    fs_dyn_da, ms_dyn_da = load_dynamic_anoms()
    fs_stat_da, ms_stat_da = load_static_anoms()
    ocean_mask = load_ocean_mask()

    # dynamic weighted means
    years_fs_dyn, fs_dyn_mean = compute_area_weighted_mean(fs_dyn_da, ocean_mask)
    years_ms_dyn, ms_dyn_mean = compute_area_weighted_mean(ms_dyn_da, ocean_mask)

    # static weighted means
    years_fs_stat, fs_stat_mean = compute_area_weighted_mean(fs_stat_da, ocean_mask)
    years_ms_stat, ms_stat_mean = compute_area_weighted_mean(ms_stat_da, ocean_mask)

    # ensure alignment
    assert np.all(years_fs_dyn == years_ms_dyn)
    assert np.all(years_fs_stat == years_ms_stat)
    assert np.all(years_fs_dyn == years_fs_stat)

    years = years_fs_dyn

    # season duration anomalies
    dur_dyn = ms_dyn_mean - fs_dyn_mean
    dur_stat = ms_stat_mean - fs_stat_mean

    # LOESS smooth: dynamic
    _, fs_dyn_smooth = loess_smooth(years, fs_dyn_mean)
    _, ms_dyn_smooth = loess_smooth(years, ms_dyn_mean)
    _, dur_dyn_smooth = loess_smooth(years, dur_dyn)

    # LOESS smooth: static
    _, fs_stat_smooth = loess_smooth(years, fs_stat_mean)
    _, ms_stat_smooth = loess_smooth(years, ms_stat_mean)
    _, dur_stat_smooth = loess_smooth(years, dur_stat)

    # dynamic std dev for bands
    fs_dyn_std = np.nanstd(fs_dyn_mean)
    ms_dyn_std = np.nanstd(ms_dyn_mean)
    dur_dyn_std = np.nanstd(dur_dyn)

    # colors
    dyn_color = "black"
    stat_color = "#1f77b4"  # blue, colorblind-friendly
    raw_color = "0.7"

    # figure
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 6.5), sharex=True)
    fig.patch.set_facecolor("white")

    panels = [
        ("(a) Season duration anomaly (days)",
         dur_dyn, dur_dyn_smooth, dur_dyn_std,
         dur_stat, dur_stat_smooth),
        ("(b) Retreat anomaly (days)",
         ms_dyn_mean, ms_dyn_smooth, ms_dyn_std,
         ms_stat_mean, ms_stat_smooth),
        ("(c) Advance anomaly (days)",
         fs_dyn_mean, fs_dyn_smooth, fs_dyn_std,
         fs_stat_mean, fs_stat_smooth),
    ]

    for i, (ax, (title,
                 raw_dyn, smooth_dyn, sigma_dyn,
                 raw_stat, smooth_stat)) in enumerate(zip(axes, panels)):

        # DYNAMIC raw (grey)
        ax.plot(years, raw_dyn, color=raw_color, linewidth=1.2)

        # DYNAMIC ±σ band
        ax.fill_between(
            years,
            smooth_dyn - sigma_dyn,
            smooth_dyn + sigma_dyn,
            color="0.85",
            alpha=0.5,
            zorder=0,
        )

        # DYNAMIC smooth
        dyn_line, = ax.plot(years, smooth_dyn, color=dyn_color, linewidth=1.8)

        # STATIC smooth
        stat_line, = ax.plot(years, smooth_stat, color=stat_color, linewidth=1.5)

        # highlight years (dynamic raw)
        for yy in YEARS_HI:
            if yy in years:
                val = raw_dyn[np.where(years == yy)][0]
                ax.scatter(yy, val, color="red", s=35, zorder=5)

        # reference line at 2016
        if 2016 in years:
            ax.axvline(2016, color="red", linestyle="--",
                       linewidth=1.2, alpha=0.7)

        ax.set_ylabel("Days", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(alpha=0.25)

        # legend only on top panel
        if i == 0:
            ax.legend(
                [dyn_line, stat_line],
                ["Dynamic", "Static"],
                loc="upper left",
                frameon=False,
                fontsize=8,
            )

    axes[-1].set_xlabel("Year", fontsize=9)

    fig.subplots_adjust(left=0.1, right=0.95, top=0.94,
                        bottom=0.08, hspace=0.35)

    # SAVE + RCLONE
    outpath = get_fig_path(
        project_root=PROJECT_ROOT,
        subfolder="anomalies",
        fig_name="Fig_FS_MS_dynamic_static_anomaly_timeseries.png",
    )

    save_and_upload(
        fig,
        outpath,
        remote_root="gdrive:sea-ice-phase/Results/Ch2_Figures",
        remote_subdir="anomalies",
    )


if __name__ == "__main__":
    plot_anomaly_timeseries()
