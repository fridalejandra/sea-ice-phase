#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Daily SIC persistence (lag-1 autocorrelation) in melt season:
  pre vs post regimes.

Regimes:
  pre  = 1980–2015
  post = 2016–2023

Season:
  Melt/retreat season: November–February (11, 12, 1, 2)

Data:
  /user/geog/falejandraperez/sea-ice-phase/data/merged/merged_bootstrap_SH_latest.nc
  variable: N07_ICECON (0–1, sea_ice_area_fraction)

Metric:

  SIC_pct = 100 * N07_ICECON

  For each regime and gridcell, in Nov–Feb:

    P = corr( SIC(t), SIC(t+1 day) )

  computed over all daily values in the regime, restricted to these months.

  Then:

    P_pre  = lag-1 corr in 1980–2015
    P_post = lag-1 corr in 2016–2023
    ΔP     = P_post - P_pre

Figure:
  Row 1:
    (a) P_pre (Nov–Feb)
    (b) P_post (Nov–Feb)
    (c) ΔP (post − pre)

  Row 2:
    (d) Sector-mean ΔP (bars)
"""

import sys
from pathlib import Path

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ---------------------------------------------------------------------
# Shared utilities
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

SIC_FILE = Path(
    "/user/geog/falejandraperez/sea-ice-phase/data/merged/merged_bootstrap_SH_latest.nc"
)
sic_var = "N07_ICECON"

SECTOR_FILE = PROJECT_ROOT / "data" / "canonical_sectors.nc"
REMOTE_ROOT = "gdrive:sea-ice-phase/Results/Ch2_Figures"
SUBFOLDER = "persistence_melt"

# Regimes
PRE_START = "1980-01-01"
PRE_END   = "2015-12-31"
POST_START = "2016-01-01"
POST_END   = "2023-12-31"

# Melt/retreat season
SEASON_MONTHS = [11, 12, 1, 2]

sector_ids = [1, 2, 3, 4, 5]
sector_labels = {
    1: "A–B",
    2: "WED",
    3: "KHV",
    4: "EA",
    5: "RA",
}

# color scale for P (0–1) and ΔP
VMIN_P = 0.0
VMAX_P = 1.0
VMAX_DP = 0.4   # tweak after first look


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------
def load_sic_and_masks():
    ds = xr.open_dataset(SIC_FILE)
    da_sic = ds[sic_var]

    if not np.issubdtype(da_sic["time"].dtype, np.datetime64):
        ds = xr.decode_cf(ds)
        da_sic = ds[sic_var]

    # convert to percent
    da_sic = da_sic * 100.0
    da_sic.name = "SIC_pct"

    ds_mask = xr.open_dataset(SECTOR_FILE)
    valid_ocean = ds_mask["valid_ocean"].astype(bool)
    sector_mask = ds_mask["sector_id"]
    ds_mask.close()

    return da_sic, valid_ocean, sector_mask


# ---------------------------------------------------------------------
# Persistence computation
# ---------------------------------------------------------------------
def lag1_corr(da):
    """
    Compute lag-1 autocorrelation along 'time' for an xarray DataArray.

    P = corr( X_t, X_{t+1} )

    Assumes time is the first dimension, but works in general as long as
    'time' is present.
    """
    x = da
    y = da.shift(time=-1)

    # drop last time step to align
    x = x.isel(time=slice(0, -1))
    y = y.isel(time=slice(0, -1))

    # mean over time
    x_mean = x.mean("time", skipna=True)
    y_mean = y.mean("time", skipna=True)

    xm = x - x_mean
    ym = y - y_mean

    cov = (xm * ym).mean("time", skipna=True)
    varx = (xm * xm).mean("time", skipna=True)
    vary = (ym * ym).mean("time", skipna=True)

    denom = np.sqrt(varx * vary)
    corr = cov / denom

    return corr


def compute_persistence(da_sic_pct):
    """
    Compute P_pre, P_post, ΔP (lag-1 correlation) for melt season.
    """

    # subset by regime
    sic_pre = da_sic_pct.sel(time=slice(PRE_START, PRE_END))
    sic_post = da_sic_pct.sel(time=slice(POST_START, POST_END))

    # restrict to melt/retreat season months
    sic_pre = sic_pre.sel(time=sic_pre["time"].dt.month.isin(SEASON_MONTHS))
    sic_post = sic_post.sel(time=sic_post["time"].dt.month.isin(SEASON_MONTHS))

    # compute lag-1 autocorrelation
    P_pre = lag1_corr(sic_pre)
    P_post = lag1_corr(sic_post)

    dP = P_post - P_pre

    P_pre.name = "P_pre"
    P_post.name = "P_post"
    dP.name = "delta_P"

    return P_pre, P_post, dP


def sector_mean_delta(dP, sector_mask, valid_ocean):
    records = []
    for sec in sector_ids:
        mask = (sector_mask == sec) & valid_ocean
        vals = dP.where(mask).values
        mean_val = float(np.nanmean(vals))

        records.append(
            {
                "sector_id": sec,
                "sector_label": sector_labels[sec],
                "delta_P": mean_val,
            }
        )

    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------
def make_polar_ax(fig, gs, row, col):
    proj = ccrs.SouthPolarStereo()
    ax = fig.add_subplot(gs[row, col], projection=proj)
    ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="0.8", edgecolor="none", zorder=2)
    ax.set_facecolor("white")
    ax.gridlines(draw_labels=False, linewidth=0.0)
    return ax


# ---------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------
def make_persistence_figure():
    sic_pct, valid_ocean, sector_mask = load_sic_and_masks()
    P_pre, P_post, dP = compute_persistence(sic_pct)

    P_pre = P_pre.where(valid_ocean)
    P_post = P_post.where(valid_ocean)
    dP = dP.where(valid_ocean)

    df_sector = sector_mean_delta(dP, sector_mask, valid_ocean)

    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(
        2, 3, height_ratios=[1, 1], width_ratios=[1.3, 1.3, 1.0], wspace=0.25, hspace=0.25
    )

    cmap_seq = plt.get_cmap("Blues")
    cmap_div = plt.get_cmap("RdBu_r")

    season_str = "Nov–Feb"

    # Row 1
    ax_pre = make_polar_ax(fig, gs, 0, 0)
    im_pre = ax_pre.pcolormesh(
        P_pre["x"],
        P_pre["y"],
        P_pre,
        transform=ccrs.SouthPolarStereo(),
        cmap=cmap_seq,
        vmin=VMIN_P,
        vmax=VMAX_P,
        shading="auto",
    )
    ax_pre.set_title(f"(a) Persistence P_pre, {season_str}", fontsize=9, fontweight="bold")

    ax_post = make_polar_ax(fig, gs, 0, 1)
    im_post = ax_post.pcolormesh(
        P_post["x"],
        P_post["y"],
        P_post,
        transform=ccrs.SouthPolarStereo(),
        cmap=cmap_seq,
        vmin=VMIN_P,
        vmax=VMAX_P,
        shading="auto",
    )
    ax_post.set_title(f"(b) Persistence P_post, {season_str}", fontsize=9, fontweight="bold")

    ax_dP = make_polar_ax(fig, gs, 0, 2)
    im_dP = ax_dP.pcolormesh(
        dP["x"],
        dP["y"],
        dP,
        transform=ccrs.SouthPolarStereo(),
        cmap=cmap_div,
        vmin=-VMAX_DP,
        vmax=VMAX_DP,
        shading="auto",
    )
    ax_dP.set_title(f"(c) ΔP (post − pre), {season_str}", fontsize=9, fontweight="bold")

    # Row 2: sector barplot
    ax_bar = fig.add_subplot(gs[1, :])
    x_positions = np.arange(len(sector_ids))
    ax_bar.bar(x_positions, df_sector["delta_P"].values, color="#4daf4a")
    ax_bar.axhline(0, color="0.2", linewidth=0.8)
    ax_bar.set_xticks(x_positions)
    ax_bar.set_xticklabels([sector_labels[s] for s in sector_ids])
    ax_bar.set_ylabel("ΔP (post − pre)")
    ax_bar.set_title(f"(d) Sector-mean ΔP, {season_str}", fontweight="bold", fontsize=9)

    max_abs = np.nanmax(np.abs(df_sector["delta_P"].values))
    if np.isfinite(max_abs) and max_abs > 0:
        ax_bar.set_ylim(-1.2 * max_abs, 1.2 * max_abs)

    # Colorbars
    cax1 = fig.add_axes([0.10, 0.06, 0.30, 0.02])
    cb1 = fig.colorbar(im_pre, cax=cax1, orientation="horizontal")
    cb1.set_label("Lag-1 persistence P (pre/post)", fontsize=9)
    cb1.ax.tick_params(labelsize=8)
    cb1.outline.set_visible(False)

    cax2 = fig.add_axes([0.55, 0.06, 0.30, 0.02])
    cb2 = fig.colorbar(im_dP, cax=cax2, orientation="horizontal")
    cb2.set_label("ΔP (post − pre)", fontsize=9)
    cb2.ax.tick_params(labelsize=8)
    cb2.outline.set_visible(False)

    fig.tight_layout(rect=[0.02, 0.10, 0.98, 0.98])

    fig_name = "Fig_SIC_persistence_melt_pre1980-2015_post2016-2023.png"
    out_path = get_fig_path(
        PROJECT_ROOT,
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
    make_persistence_figure()


if __name__ == "__main__":
    main()
