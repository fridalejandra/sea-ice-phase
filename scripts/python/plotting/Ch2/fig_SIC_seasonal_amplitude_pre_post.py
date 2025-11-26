#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Seasonal SIC amplitude: pre vs post (1980–2017 vs 2018–2023).

Data:
  /user/geog/falejandraperez/sea-ice-phase/data/merged/merged_bootstrap_SH_latest.nc
  variable: N07_ICECON (0–1)

Metric:

  Convert to percent:
    SIC_pct = 100 * N07_ICECON

  For each year and gridcell:
    Amp(year) = max(SIC_pct) - min(SIC_pct)   [%-points over calendar year]

  Then:
    Amp_pre  = mean Amp over 1980–2017
    Amp_post = mean Amp over 2018–2023
    ΔAmp     = Amp_post - Amp_pre             [%-points]

Figure:
  Row 1:
    (a) Amp_pre
    (b) Amp_post
    (c) ΔAmp

  Row 2:
    (d) Sector-mean ΔAmp
"""

import sys
from pathlib import Path

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

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
SUBFOLDER = "amplitude"

PRE_START_YEAR = 1980
PRE_END_YEAR = 2017
POST_START_YEAR = 2018
POST_END_YEAR = 2023

sector_ids = [1, 2, 3, 4, 5]
sector_labels = {
    1: "A–B",
    2: "WED",
    3: "KHV",
    4: "EA",
    5: "RA",
}


def load_sic_and_masks():
    ds = xr.open_dataset(SIC_FILE)
    da_sic = ds[sic_var]

    if not np.issubdtype(da_sic["time"].dtype, np.datetime64):
        ds = xr.decode_cf(ds)
        da_sic = ds[sic_var]

    # to percent
    da_sic = da_sic * 100.0
    da_sic.name = "SIC_pct"

    ds_mask = xr.open_dataset(SECTOR_FILE)
    valid_ocean = ds_mask["valid_ocean"].astype(bool)
    sector_mask = ds_mask["sector_id"]
    ds_mask.close()

    return da_sic, valid_ocean, sector_mask


def compute_annual_amplitude(da_sic_pct):
    """
    Amp(year, y, x) = max(SIC_pct) - min(SIC_pct) over the calendar year.
    """
    ds_year = da_sic_pct.groupby("time.year")
    sic_max = ds_year.max("time", skipna=True)
    sic_min = ds_year.min("time", skipna=True)

    amp = sic_max - sic_min
    amp.name = "annual_amp_pct"
    return amp


def compute_pre_post_amp(amp):
    amp_pre = amp.sel(year=slice(PRE_START_YEAR, PRE_END_YEAR)).mean(
        "year", skipna=True
    )
    amp_post = amp.sel(year=slice(POST_START_YEAR, POST_END_YEAR)).mean(
        "year", skipna=True
    )
    delta_amp = amp_post - amp_pre

    amp_pre.name = "amp_pre"
    amp_post.name = "amp_post"
    delta_amp.name = "delta_amp"

    return amp_pre, amp_post, delta_amp


def sector_mean_delta_amp(delta_amp, sector_mask, valid_ocean):
    records = []
    for sec in sector_ids:
        mask = (sector_mask == sec) & valid_ocean
        vals = delta_amp.where(mask).values
        mean_val = float(np.nanmean(vals))

        records.append(
            {
                "sector_id": sec,
                "sector_label": sector_labels[sec],
                "delta_amp": mean_val,
            }
        )
    return pd.DataFrame.from_records(records)


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


def make_amplitude_figure():
    sic_pct, valid_ocean, sector_mask = load_sic_and_masks()
    amp = compute_annual_amplitude(sic_pct)
    amp_pre, amp_post, delta_amp = compute_pre_post_amp(amp)

    amp_pre = amp_pre.where(valid_ocean)
    amp_post = amp_post.where(valid_ocean)
    delta_amp = delta_amp.where(valid_ocean)

    df_sector = sector_mean_delta_amp(delta_amp, sector_mask, valid_ocean)

    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(
        2, 3, height_ratios=[1, 1], width_ratios=[1.3, 1.3, 1.0], wspace=0.25, hspace=0.25
    )

    cmap_seq = plt.get_cmap("PuBu")
    cmap_div = plt.get_cmap("RdBu_r")

    vmax_amp = float(np.nanmax(amp_pre.values))
    # cap at 100% just in case
    vmax_amp = min(vmax_amp, 100.0)

    vmax_delta = float(np.nanmax(np.abs(delta_amp.values)))
    vmax_delta = min(vmax_delta, 60.0)  # tweak if needed

    # Row 1: Amp_pre, Amp_post, ΔAmp
    ax_pre = make_polar_ax(fig, gs, 0, 0)
    im_pre = ax_pre.pcolormesh(
        amp_pre["x"],
        amp_pre["y"],
        amp_pre,
        transform=ccrs.SouthPolarStereo(),
        cmap=cmap_seq,
        vmin=0.0,
        vmax=vmax_amp,
        shading="auto",
    )
    ax_pre.set_title("(a) Seasonal amplitude, pre (1980–2017) [%]", fontsize=9, fontweight="bold")

    ax_post = make_polar_ax(fig, gs, 0, 1)
    im_post = ax_post.pcolormesh(
        amp_post["x"],
        amp_post["y"],
        amp_post,
        transform=ccrs.SouthPolarStereo(),
        cmap=cmap_seq,
        vmin=0.0,
        vmax=vmax_amp,
        shading="auto",
    )
    ax_post.set_title("(b) Seasonal amplitude, post (2018–2023) [%]", fontsize=9, fontweight="bold")

    ax_delta = make_polar_ax(fig, gs, 0, 2)
    im_delta = ax_delta.pcolormesh(
        delta_amp["x"],
        delta_amp["y"],
        delta_amp,
        transform=ccrs.SouthPolarStereo(),
        cmap=cmap_div,
        vmin=-vmax_delta,
        vmax=vmax_delta,
        shading="auto",
    )
    ax_delta.set_title("(c) ΔAmplitude (post − pre) [%]", fontsize=9, fontweight="bold")

    # Row 2: sector barplot
    ax_bar = fig.add_subplot(gs[1, :])
    x_positions = np.arange(len(sector_ids))
    ax_bar.bar(x_positions, df_sector["delta_amp"].values, color="#984ea3")
    ax_bar.axhline(0, color="0.2", linewidth=0.8)
    ax_bar.set_xticks(x_positions)
    ax_bar.set_xticklabels([sector_labels[s] for s in sector_ids])
    ax_bar.set_ylabel("ΔAmplitude (post − pre) [%]")
    ax_bar.set_title("(d) Sector-mean ΔAmplitude", fontweight="bold", fontsize=9)

    max_abs = np.nanmax(np.abs(df_sector["delta_amp"].values))
    if np.isfinite(max_abs) and max_abs > 0:
        ax_bar.set_ylim(-1.2 * max_abs, 1.2 * max_abs)

    # Colorbars
    cax1 = fig.add_axes([0.10, 0.06, 0.30, 0.02])
    cb1 = fig.colorbar(im_pre, cax=cax1, orientation="horizontal")
    cb1.set_label("Seasonal amplitude of SIC [%]", fontsize=9)
    cb1.ax.tick_params(labelsize=8)
    cb1.outline.set_visible(False)

    cax2 = fig.add_axes([0.55, 0.06, 0.30, 0.02])
    cb2 = fig.colorbar(im_delta, cax=cax2, orientation="horizontal")
    cb2.set_label("ΔAmplitude (post − pre) [%]", fontsize=9)
    cb2.ax.tick_params(labelsize=8)
    cb2.outline.set_visible(False)

    fig.tight_layout(rect=[0.02, 0.10, 0.98, 0.98])

    fig_name = "Fig_SIC_seasonal_amplitude_pre1980-2017_post2018-2023.png"
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
    make_amplitude_figure()


if __name__ == "__main__":
    main()
