#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fig_FS_MS_anomaly_timeseries_dynamic.py

Handcock-style anomaly time series:
  (a) season duration anomaly (MS − FS)
  (b) MS anomaly (retreat)
  (c) FS anomaly (advance)

Uses dynamic anomalies only:
  results/anomalies/FS_dynamic_anomalies.nc
  results/anomalies/MS_dynamic_anomalies.nc

Features:
  - area-weighted mean anomalies
  - raw yearly curves (thin grey)
  - LOESS-smoothed curve
  - ±1σ band
  - highlight 2016, 2022, 2023
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


# highlight years
YEARS_HI = [2016, 2022, 2023]

# ---------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------

def load_dynamic_anoms():
    fs = xr.open_dataset(ANOM_DIR / "FS_dynamic_anomalies.nc")["FS_dynamic_anom"]
    ms = xr.open_dataset(ANOM_DIR / "MS_dynamic_anomalies.nc")["MS_dynamic_anom"]
    return fs, ms

def load_ocean_mask():
    ds = xr.open_dataset(SECTOR_FILE)
    # valid ocean mask already provided
    return ds["valid_ocean"].astype(bool).values

# ---------------------------------------------------------------------
# AREA-WEIGHTED MEAN
# ---------------------------------------------------------------------

def area_weights_from_lat(lat):
    """Simple cosine-lat weight for polar grid."""
    return np.cos(np.deg2rad(lat))

def compute_area_weighted_mean(da, ocean_mask):
    """
    da: (year, y, x)
    ocean_mask: (y, x)
    """
    # if lat exists, compute cosine weighting, else uniform
    if "lat" in da.coords:
        lat = da["lat"].values
        w = np.cos(np.deg2rad(lat))
        # broadcast to 2D
        W = np.repeat(w[:, None], da.shape[2], axis=1)
    else:
        # fallback: uniform weights over valid ocean grid
        W = np.ones_like(ocean_mask, dtype=float)

    W = W * ocean_mask
    W = W / np.nansum(W)

    # weighted mean for each year
    out = []
    years = da["year"].values
    for y in years:
        arr = da.sel(year=y).values
        out.append(np.nansum(arr * W))

    return years, np.array(out)

# ---------------------------------------------------------------------
# LOESS SMOOTHING
# ---------------------------------------------------------------------

def loess_smooth(x, y, frac=0.25):
    """
    LOESS smoother. frac = smoothing span.
    """
    lo = sm.nonparametric.lowess(y, x, frac=frac, return_sorted=True)
    return lo[:, 0], lo[:, 1]

# ---------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------

def plot_anomaly_timeseries():
    # load anomalies
    fs_anom, ms_anom = load_dynamic_anoms()
    ocean_mask = load_ocean_mask()

    # compute weighted means
    years_fs, fs_mean = compute_area_weighted_mean(fs_anom, ocean_mask)
    years_ms, ms_mean = compute_area_weighted_mean(ms_anom, ocean_mask)

    # ensure alignment
    assert np.all(years_fs == years_ms)
    years = years_fs

    # season duration anomaly
    dur = ms_mean - fs_mean

    # smooth curves
    x_lo, fs_smooth = loess_smooth(years, fs_mean)
    _, ms_smooth = loess_smooth(years, ms_mean)
    _, dur_smooth = loess_smooth(years, dur)

    # std deviation bands
    fs_std = np.nanstd(fs_mean)
    ms_std = np.nanstd(ms_mean)
    dur_std = np.nanstd(dur)

    # figure
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 6.5), sharex=True)
    fig.patch.set_facecolor("white")

    panels = [
        ("(a) Season duration anomaly (days)", dur, dur_smooth, dur_std),
        ("(b) Retreat anomaly (days)", ms_mean, ms_smooth, ms_std),
        ("(c) Advance anomaly (days)", fs_mean, fs_smooth, fs_std),
    ]

    for ax, (title, raw, smooth, sigma) in zip(axes, panels):
        # raw
        ax.plot(years, raw, color="0.7", linewidth=1.2)

        # ±1σ band
        ax.fill_between(
            years,
            smooth - sigma,
            smooth + sigma,
            color="0.85",
            alpha=0.5,
            zorder=0,
        )

        # smooth
        ax.plot(years, smooth, color="black", linewidth=1.8)

        # highlight years
        for yy in YEARS_HI:
            if yy in years:
                val = raw[np.where(years == yy)][0]
                ax.scatter(yy, val, color="red", s=35, zorder=5)

        # vertical ref line at 2016
        if 2016 in years:
            ax.axvline(2016, color="red", linestyle="--", linewidth=1.2, alpha=0.7)

        ax.set_ylabel("Days", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")

        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("Year", fontsize=9)

    fig.tight_layout()

    from ch2_fig_utils import get_fig_path, save_and_upload

    outpath = get_fig_path(
        project_root=PROJECT_ROOT,
        subfolder="anomalies",
        fig_name="Fig_FS_MS_dynamic_anomaly_timeseries.png",
    )

    save_and_upload(
        fig,
        outpath,
        remote_root="gdrive:sea-ice-phase/Results/Ch2_Figures",
        remote_subdir="anomalies",
    )

if __name__ == "__main__":
    plot_anomaly_timeseries()
