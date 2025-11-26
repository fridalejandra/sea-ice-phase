#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Daily SIC volatility: pre vs post (1980–2017 vs 2018–2023).

Data:
  /user/geog/falejandraperez/sea-ice-phase/data/merged/merged_bootstrap_SH_latest.nc
  variable: N07_ICECON (sea_ice_area_fraction, 0–1)

Metric:
  1) Convert SIC to percent:
       SIC_pct = 100 * N07_ICECON

  2) Daily difference:
       dSIC(t) = SIC_pct(t) - SIC_pct(t-1)

  3) Volatility in each regime:
       sigma_pre  = std(dSIC) over 1980-01-01..2017-12-31
       sigma_post = std(dSIC) over 2018-01-01..2023-12-31

  4) Change:
       Δσ_daily = sigma_post - sigma_pre  [%-points per day]

Figure:
  Row 1:
    (a) σ_pre(dSIC)
    (b) σ_post(dSIC)
    (c) Δσ_daily (post − pre)

  Row 2:
    (d) Sector-mean Δσ_daily (bars)
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

SIC_FILE = Path(
    "/user/geog/falejandraperez/sea-ice-phase/data/merged/merged_bootstrap_SH_latest.nc"
)
sic_var = "N07_ICECON"  # 0–1, sea_ice_area_fraction

SECTOR_FILE = PROJECT_ROOT / "data" / "canonical_sectors.nc"

REMOTE_ROOT = "gdrive:sea-ice-phase/Results/Ch2_Figures"
SUBFOLDER = "volatility_daily"

PRE_START = "1980-01-01"
PRE_END = "2017-12-31"
POST_START = "2018-01-01"
POST_END = "2023-12-31"

sector_ids = [1, 2, 3, 4, 5]
sector_labels = {
    1: "A–B",
    2: "WED",
    3: "KHV",
    4: "EA",
    5: "RA",
}

# expected σ(dSIC) ~ a few %-points; adjust if needed
VMAX_SIG = 10.0  # [% per day]


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------
def load_sic_and_masks():
    ds = xr.open_dataset(SIC_FILE)
    da_sic = ds[sic_var]

    # decode time if needed
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
# Volatility computation
# ---------------------------------------------------------------------
def compute_daily_diff_volatility(da_sic_pct):
    """
    Compute std of daily SIC differences (dSIC, in %-points) in pre and post periods.
    """
    # daily difference over full record
    dSIC = da_sic_pct.diff("time")  # (time-1, y, x)
    dSIC = dSIC.assign_coords(time=da_sic_pct["time"][1:])

    dSIC_pre = dSIC.sel(time=slice(PRE_START, PRE_END))
    dSIC_post = dSIC.sel(time=slice(POST_START, POST_END))

    sigma_pre = dSIC_pre.std("time", skipna=True)
    sigma_post = dSIC_post.std("time", skipna=True)

    delta_sigma = sigma_post - sigma_pre

    sigma_pre.name = "sigma_pre_daily"
    sigma_post.name = "sigma_post_daily"
    delta_sigma.name = "delta_sigma_daily"

    return sigma_pre, sigma_post, delta_sigma


def sector_mean_delta(delta_sigma, sector_mask, valid_ocean):
    """
    Sector-mean Δσ_daily.
    """
    records = []
    for sec in sector_ids:
        mask = (sector_mask == sec) & valid_ocean
        vals = delta_sigma.where(mask).values
        mean_val = float(np.nanmean(vals))

        records.append(
            {
                "sector_id": sec,
                "sector_label": sector_labels[sec],
                "delta_sigma": mean_val,
            }
        )

    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------
def make_polar_ax(fig, gs, row, col):
    proj = ccrs.SouthPolarStereo()
    ax = fig.add_subplot(gs[row, col], projection=proj)
    ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="0.8", edgecolor="none", zorder=2)
    ax.set_facecolor("white")
    ax.gridlines(
        draw_labels=False,
        linewidth=0.0,
        color="0.7",
        alpha=0.4,
        linestyle="--",
    )
    return ax


# ---------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------
def make_daily_volatility_figure():
    sic_pct, valid_ocean, sector_mask = load_sic_and_masks()
    sigma_pre, sigma_post, delta_sigma = compute_daily_diff_volatility(sic_pct)

    sigma_pre = sigma_pre.where(valid_ocean)
    sigma_post = sigma_post.where(valid_ocean)
    delta_sigma = delta_sigma.where(valid_ocean)

    df_sector = sector_mean_delta(delta_sigma, sector_mask, valid_ocean)

    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(
        2, 3, height_ratios=[1, 1], width_ratios=[1.3, 1.3, 1.0], wspace=0.25, hspace=0.25
    )

    cmap_seq = plt.get_cmap("Blues")
    cmap_div = plt.get_cmap("RdBu_r")

    # Row 1: σ_pre, σ_post, Δσ
    ax_pre = make_polar_ax(fig, gs, 0, 0)
    im_pre = ax_pre.pcolormesh(
        sigma_pre["x"],
        sigma_pre["y"],
        sigma_pre,
        transform=ccrs.SouthPolarStereo(),
        cmap=cmap_seq,
        vmin=0.0,
        vmax=VMAX_SIG,
        shading="auto",
    )
    ax_pre.set_title("(a) σ_pre(dSIC) [%/day]", fontsize=9, fontweight="bold")

    ax_post = make_polar_ax(fig, gs, 0, 1)
    im_post = ax_post.pcolormesh(
        sigma_post["x"],
        sigma_post["y"],
        sigma_post,
        transform=ccrs.SouthPolarStereo(),
        cmap=cmap_seq,
        vmin=0.0,
        vmax=VMAX_SIG,
        shading="auto",
    )
    ax_post.set_title("(b) σ_post(dSIC) [%/day]", fontsize=9, fontweight="bold")

    ax_delta = make_polar_ax(fig, gs, 0, 2)
    im_delta = ax_delta.pcolormesh(
        delta_sigma["x"],
        delta_sigma["y"],
        delta_sigma,
        transform=ccrs.SouthPolarStereo(),
        cmap=cmap_div,
        vmin=-VMAX_SIG,
        vmax=VMAX_SIG,
        shading="auto",
    )
    ax_delta.set_title("(c) Δσ_daily (post − pre) [%/day]", fontsize=9, fontweight="bold")

    # Row 2: sector barplot
    ax_bar = fig.add_subplot(gs[1, :])
    x_positions = np.arange(len(sector_ids))
    ax_bar.bar(x_positions, df_sector["delta_sigma"].values, color="#377eb8")
    ax_bar.axhline(0, color="0.2", linewidth=0.8)
    ax_bar.set_xticks(x_positions)
    ax_bar.set_xticklabels([sector_labels[s] for s in sector_ids])
    ax_bar.set_ylabel("Δσ_daily (post − pre) [%/day]")
    ax_bar.set_title("(d) Sector-mean Δσ_daily", fontweight="bold", fontsize=9)

    # y-limits symmetric around 0
    max_abs = np.nanmax(np.abs(df_sector["delta_sigma"].values))
    if np.isfinite(max_abs) and max_abs > 0:
        ax_bar.set_ylim(-1.2 * max_abs, 1.2 * max_abs)

    # Colorbars
    cax1 = fig.add_axes([0.10, 0.06, 0.30, 0.02])
    cb1 = fig.colorbar(im_pre, cax=cax1, orientation="horizontal")
    cb1.set_label("σ(dSIC) [%/day] (pre/post)", fontsize=9)
    cb1.ax.tick_params(labelsize=8)
    cb1.outline.set_visible(False)

    cax2 = fig.add_axes([0.55, 0.06, 0.30, 0.02])
    cb2 = fig.colorbar(im_delta, cax=cax2, orientation="horizontal")
    cb2.set_label("Δσ_daily (post − pre) [%/day]", fontsize=9)
    cb2.ax.tick_params(labelsize=8)
    cb2.outline.set_visible(False)

    fig.tight_layout(rect=[0.02, 0.10, 0.98, 0.98])

    fig_name = "Fig_SIC_daily_volatility_pre1980-2017_post2018-2023.png"
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
    make_daily_volatility_figure()


if __name__ == "__main__":
    main()
