#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fig_FS_MS_anomaly_timeseries_dynamic_static_with_seasonal_mask.py

Compare Antarctic-mean FS/MS anomalies:
  - dynamic vs static
  - with vs without 'seasonal ice zone' mask.

Inputs (must already exist):
  results/anomalies/FS_dynamic_anomalies.nc  (var: FS_dynamic_anom)
  results/anomalies/MS_dynamic_anomalies.nc  (var: MS_dynamic_anom)
  results/anomalies/FS_static_anomalies.nc   (var: FS_static_anom)
  results/anomalies/MS_static_anomalies.nc   (var: MS_static_anom)

  data/canonical_sectors.nc    (for valid_ocean mask)
  results/masks/seasonal_ice_zone_mask.nc (from make_seasonal_ice_zone_mask.py)

Method:
  - Compute area-weighted mean anomalies using:
        (a) valid_ocean only           → "All ice zone"
        (b) valid_ocean & seasonal_mask → "Seasonal ice zone"
  - Apply the same cosine(lat) weighting in both cases.
  - Plot Handcock-style three-panel figure:
        (a) season duration anomaly  (MS - FS)
        (b) MS anomaly (retreat)
        (c) FS anomaly (advance)
    with:
        • thin grey: yearly values (All ice zone)
        • black solid line: LOESS (Seasonal ice zone)
        • blue solid line: LOESS (All ice zone)
        • grey band: ±1σ for Seasonal ice zone
        • red points: highlighted years (2016, 2022, 2023)
    Vertical dashed line at 2016.

This will show directly how much the seasonal ice zone mask
cleans up the SMMR-era artifacts in the early 1980s.
"""

import sys
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import statsmodels.api as sm

# ---------------------------------------------------------------------
# Add plotting root to path and import Ch2 utilities
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

ANOM_DIR   = PROJECT_ROOT / "results" / "anomalies"
SECTOR_FILE = PROJECT_ROOT / "data" / "canonical_sectors.nc"
MASK_FILE   = PROJECT_ROOT / "results" / "masks" / "seasonal_ice_zone_mask.nc"

YEARS_HI = [2016, 2022, 2023]

# Year range to plot (you can adjust)
YEAR_MIN = 1980
YEAR_MAX = 2023


# ---------------------------------------------------------------------
# DATA LOADERS
# ---------------------------------------------------------------------

def load_anoms():
    fs_dyn = xr.open_dataset(ANOM_DIR / "FS_dynamic_anomalies.nc")["FS_dynamic_anom"]
    ms_dyn = xr.open_dataset(ANOM_DIR / "MS_dynamic_anomalies.nc")["MS_dynamic_anom"]

    fs_sta = xr.open_dataset(ANOM_DIR / "FS_static_anomalies.nc")["FS_static_anom"]
    ms_sta = xr.open_dataset(ANOM_DIR / "MS_static_anomalies.nc")["MS_static_anom"]

    return fs_dyn, ms_dyn, fs_sta, ms_sta


def load_masks():
    ds = xr.open_dataset(SECTOR_FILE)
    ocean_mask = ds["valid_ocean"].astype(bool)

    mask_seasonal = xr.open_dataset(MASK_FILE)["seasonal_ice_zone"].astype(bool)

    # align
    mask_seasonal = mask_seasonal.broadcast_like(ocean_mask)
    return ocean_mask.values, mask_seasonal.values


def area_weights(lat):
    """cos(lat) weights, normalized over valid mask later."""
    return np.cos(np.deg2rad(lat))


def compute_weighted_mean(da, base_mask, seasonal_mask=None):
    """
    da: DataArray (year, y, x)
    base_mask: (y, x) valid_ocean
    seasonal_mask: (y, x) or None
        If None: use base_mask only (All ice zone)
        Else: use base_mask & seasonal_mask (Seasonal ice zone)
    """
    da = da.sel(year=slice(YEAR_MIN, YEAR_MAX))

    if "lat" in da.coords:
        lat = da["lat"].values
        W = area_weights(lat)
        if W.ndim == 1:
            # lat(y) → broadcast to (y,x)
            W = np.repeat(W[:, None], da.shape[2], axis=1)
    else:
        # uniform weights
        W = np.ones_like(base_mask, dtype=float)

    mask_full = base_mask.copy()
    if seasonal_mask is not None:
        mask_full = mask_full & seasonal_mask

    W = W * mask_full
    W = W / np.nansum(W)

    years = da["year"].values
    out = []
    for y in years:
        arr = da.sel(year=y).values
        out.append(np.nansum(arr * W))

    return years, np.array(out)


def loess_smooth(x, y, frac=0.25):
    lo = sm.nonparametric.lowess(y, x, frac=frac, return_sorted=True)
    return lo[:, 0], lo[:, 1]


# ---------------------------------------------------------------------
# MAIN PLOTTING
# ---------------------------------------------------------------------

def main():
    fs_dyn, ms_dyn, fs_sta, ms_sta = load_anoms()
    ocean_mask, seasonal_mask = load_masks()

    # We will only use the dynamic anomalies for the Handcock-style figure;
    # static is mostly for side-by-side comparison if needed later.
    fs_anom = fs_dyn
    ms_anom = ms_dyn

    # All-ice-zone and seasonal-ice-zone means
    years_all, fs_all = compute_weighted_mean(fs_anom, ocean_mask, seasonal_mask=None)
    years_all2, ms_all = compute_weighted_mean(ms_anom, ocean_mask, seasonal_mask=None)

    years_sea, fs_sea = compute_weighted_mean(fs_anom, ocean_mask, seasonal_mask=seasonal_mask)
    years_sea2, ms_sea = compute_weighted_mean(ms_anom, ocean_mask, seasonal_mask=seasonal_mask)

    # Sanity alignment
    assert np.all(years_all == years_all2)
    assert np.all(years_sea == years_sea2)
    assert np.all(years_all == years_sea)

    years = years_all

    # Duration anomalies
    dur_all = ms_all - fs_all
    dur_sea = ms_sea - fs_sea

    # LOESS
    x_lo, fs_all_s  = loess_smooth(years, fs_all)
    _,    fs_sea_s  = loess_smooth(years, fs_sea)
    _,    ms_all_s  = loess_smooth(years, ms_all)
    _,    ms_sea_s  = loess_smooth(years, ms_sea)
    _,    dur_all_s = loess_smooth(years, dur_all)
    _,    dur_sea_s = loess_smooth(years, dur_sea)

    # σ bands from seasonal-ice-zone series
    fs_std  = np.nanstd(fs_sea)
    ms_std  = np.nanstd(ms_sea)
    dur_std = np.nanstd(dur_sea)

    fig, axes = plt.subplots(3, 1, figsize=(7.5, 6.5), sharex=True)
    fig.patch.set_facecolor("white")

    panels = [
        ("(a) Season duration anomaly (days)", dur_all, dur_all_s, dur_sea_s, dur_std),
        ("(b) Retreat anomaly (days)",        ms_all, ms_all_s, ms_sea_s, ms_std),
        ("(c) Advance anomaly (days)",        fs_all, fs_all_s, fs_sea_s, fs_std),
    ]

    for ax, (title, raw_all, smooth_all, smooth_sea, sigma) in zip(axes, panels):
        # Thin grey: yearly all-ice-zone values
        ax.plot(years, raw_all, color="0.7", linewidth=1.0, zorder=1)

        # ±1σ band for seasonal mask
        ax.fill_between(
            years,
            smooth_sea - sigma,
            smooth_sea + sigma,
            color="0.85",
            alpha=0.7,
            zorder=0,
        )

        # LOESS lines
        line_all, = ax.plot(years, smooth_all, color="#4C72B0", linewidth=1.6, label="All ice zone")   # blue
        line_sea, = ax.plot(years, smooth_sea, color="black",   linewidth=1.8, label="Seasonal ice zone")

        # Highlight years (using seasonal series)
        for yy in YEARS_HI:
            if yy in years:
                idx = np.where(years == yy)[0][0]
                val = smooth_sea[idx]
                ax.scatter(yy, val, color="red", s=30, zorder=5)

        # Vertical ref line at 2016
        if 2016 in years:
            ax.axvline(2016, color="red", linestyle="--", linewidth=1.2, alpha=0.7)

        ax.set_ylabel("Days", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("Year", fontsize=9)

    # Legend in top panel
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, loc="upper left", frameon=False)

    fig.tight_layout()

    outpath = get_fig_path(
        project_root=PROJECT_ROOT,
        subfolder="anomalies",
        fig_name="Fig_FS_MS_dynamic_anomaly_timeseries_with_seasonal_mask.png",
    )

    save_and_upload(
        fig,
        outpath,
        remote_root="gdrive:sea-ice-phase/Results/Ch2_Figures",
        remote_subdir="anomalies",
    )


if __name__ == "__main__":
    main()
