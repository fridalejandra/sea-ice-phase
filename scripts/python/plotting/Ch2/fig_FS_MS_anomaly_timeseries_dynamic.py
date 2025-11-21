#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fig_FS_MS_anomaly_timeseries_dynamic_static_pre2016baseline.py

Handcock-style anomaly time series with BOTH dynamic and static,
using a PRE-2016 baseline (1980–2016), following Himmich et al.:

  (a) Season duration anomaly (MS − FS)
  (b) MS anomaly (retreat)
  (c) FS anomaly (advance)

Inputs (already computed elsewhere, any baseline):
    results/anomalies/FS_dynamic_anomalies.nc  -> FS_dynamic_anom(year,y,x)
    results/anomalies/MS_dynamic_anomalies.nc  -> MS_dynamic_anom(year,y,x)
    results/anomalies/FS_static_anomalies.nc   -> FS_static_anom(year,y,x)
    results/anomalies/MS_static_anomalies.nc   -> MS_static_anom(year,y,x)

This script:
    * area-weights over valid ocean
    * re-centers each time series so that mean over 1980–2016 = 0
    * plots:
        - dynamic: raw (grey), LOESS smooth (black), ±1σ band
        - static: LOESS smooth only (blue)
    * highlights 2016, 2022, 2023
    * vertical dashed line at 2016
"""

import sys
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import statsmodels.api as sm

# ---------------------------------------------------------------------
# ch2_fig_utils + project root
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

PROJECT_ROOT = PROJECT_ROOT_CLUSTER
ANOM_DIR = PROJECT_ROOT / "results" / "anomalies"
SECTOR_FILE = PROJECT_ROOT / "data" / "canonical_sectors.nc"

# baseline period for re-centering
BASELINE_START = 1980
BASELINE_END = 2016

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
    da: DataArray (year, y, x)
    ocean_mask: (y, x)
    Returns years, 1D array of area-weighted means (whatever the field is).
    """
    years = da["year"].values

    # if lat dimension exists, use cos(lat), else uniform
    if "lat" in da.coords:
        lat = da["lat"].values
        w1d = np.cos(np.deg2rad(lat))
        W = np.repeat(w1d[:, None], da.shape[2], axis=1)
    else:
        W = np.ones_like(ocean_mask, dtype=float)

    W = W * ocean_mask
    W = W / np.nansum(W)

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
# REBASELINE TO 1980–2016
# ---------------------------------------------------------------------
def rebaseline_to_pre2016(years, series):
    """
    Given a time series (could already be anomalies),
    subtract its mean over BASELINE_START–BASELINE_END.

    This is equivalent to recomputing anomalies relative to
    the 1980–2016 mean of the underlying dates.
    """
    mask = (years >= BASELINE_START) & (years <= BASELINE_END)
    if not np.any(mask):
        raise ValueError("No baseline years found in series.")

    baseline_mean = np.nanmean(series[mask])
    return series - baseline_mean


# ---------------------------------------------------------------------
# MAIN PLOT
# ---------------------------------------------------------------------
def plot_anomaly_timeseries():

    fs_dyn_da, ms_dyn_da = load_dynamic_anoms()
    fs_stat_da, ms_stat_da = load_static_anoms()
    ocean_mask = load_ocean_mask()

    # area-weighted means of whatever the anomaly fields currently contain
    yrs_fs_dyn, fs_dyn_raw = compute_area_weighted_mean(fs_dyn_da, ocean_mask)
    yrs_ms_dyn, ms_dyn_raw = compute_area_weighted_mean(ms_dyn_da, ocean_mask)
    yrs_fs_stat, fs_stat_raw = compute_area_weighted_mean(fs_stat_da, ocean_mask)
    yrs_ms_stat, ms_stat_raw = compute_area_weighted_mean(ms_stat_da, ocean_mask)

    # sanity: align years
    if not (np.array_equal(yrs_fs_dyn, yrs_ms_dyn)
            and np.array_equal(yrs_fs_dyn, yrs_fs_stat)
            and np.array_equal(yrs_fs_dyn, yrs_ms_stat)):
        raise RuntimeError("Year arrays for FS/MS static/dynamic do not align.")

    years = yrs_fs_dyn

    # ---- rebaseline each to 1980–2016
    fs_dyn = rebaseline_to_pre2016(years, fs_dyn_raw)
    ms_dyn = rebaseline_to_pre2016(years, ms_dyn_raw)
    fs_stat = rebaseline_to_pre2016(years, fs_stat_raw)
    ms_stat = rebaseline_to_pre2016(years, ms_stat_raw)

    # season duration anomalies (dynamic & static)
    dur_dyn_raw = ms_dyn_raw - fs_dyn_raw
    dur_stat_raw = ms_stat_raw - fs_stat_raw

    dur_dyn = rebaseline_to_pre2016(years, dur_dyn_raw)
    dur_stat = rebaseline_to_pre2016(years, dur_stat_raw)

    # LOESS smooth (dynamic + static)
    _, fs_dyn_smooth = loess_smooth(years, fs_dyn)
    _, ms_dyn_smooth = loess_smooth(years, ms_dyn)
    _, dur_dyn_smooth = loess_smooth(years, dur_dyn)

    _, fs_stat_smooth = loess_smooth(years, fs_stat)
    _, ms_stat_smooth = loess_smooth(years, ms_stat)
    _, dur_stat_smooth = loess_smooth(years, dur_stat)

    # dynamic std devs (for bands)
    fs_dyn_std = np.nanstd(fs_dyn)
    ms_dyn_std = np.nanstd(ms_dyn)
    dur_dyn_std = np.nanstd(dur_dyn)

    dyn_color = "black"
    stat_color = "#1f77b4"  # blue; reasonably colorblind friendly
    raw_color = "0.7"

    fig, axes = plt.subplots(3, 1, figsize=(7.5, 6.5), sharex=True)
    fig.patch.set_facecolor("white")

    panels = [
        ("(a) Season duration anomaly (days)",
         dur_dyn, dur_dyn_smooth, dur_dyn_std,
         dur_stat, dur_stat_smooth),
        ("(b) Retreat anomaly (days)",
         ms_dyn, ms_dyn_smooth, ms_dyn_std,
         ms_stat, ms_stat_smooth),
        ("(c) Advance anomaly (days)",
         fs_dyn, fs_dyn_smooth, fs_dyn_std,
         fs_stat, fs_stat_smooth),
    ]

    for i, (ax, (title,
                 dyn_series, dyn_smooth, dyn_sigma,
                 stat_series, stat_smooth)) in enumerate(zip(axes, panels)):

        # dynamic raw (grey)
        ax.plot(years, dyn_series, color=raw_color, linewidth=1.2)

        # dynamic ±1σ band
        ax.fill_between(
            years,
            dyn_smooth - dyn_sigma,
            dyn_smooth + dyn_sigma,
            color="0.85",
            alpha=0.5,
            zorder=0,
        )

        # dynamic smooth
        dyn_line, = ax.plot(years, dyn_smooth, color=dyn_color, linewidth=1.8)

        # static smooth
        stat_line, = ax.plot(years, stat_smooth, color=stat_color, linewidth=1.5)

        # highlight key years on dynamic series
        for yy in YEARS_HI:
            if yy in years:
                idx = np.where(years == yy)[0][0]
                ax.scatter(yy, dyn_series[idx], color="red", s=35, zorder=5)

        # vertical line at 2016
        if BASELINE_END in years:
            ax.axvline(BASELINE_END, color="red", linestyle="--",
                       linewidth=1.2, alpha=0.7)

        ax.set_ylabel("Days", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(alpha=0.25)

        # legend only on first panel
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

    # save + upload
    outpath = get_fig_path(
        project_root=PROJECT_ROOT,
        subfolder="anomalies",
        fig_name="Fig_FS_MS_dynamic_static_anomaly_timeseries_pre2016baseline.png",
    )

    save_and_upload(
        fig,
        outpath,
        remote_root="gdrive:sea-ice-phase/Results/Ch2_Figures",
        remote_subdir="anomalies",
    )


if __name__ == "__main__":
    plot_anomaly_timeseries()
