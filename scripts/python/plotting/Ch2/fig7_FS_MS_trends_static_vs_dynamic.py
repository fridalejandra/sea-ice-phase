#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FS/MS change comparison: static vs dynamic (post-2017 step change focus).

2x3 figure:
  Col 1: Pre–post SIGN/AGREEMENT CLASS map (post mean − pre mean)
  Col 2: Sector-mean (post − pre) barplots (Static vs Dynamic)
  Col 3: Magnitude difference map: (post−pre)_dynamic − (post−pre)_static  [days]

Rows:
  Row 1: FS  (calendar DOY)
  Row 2: MS  (days since Aug 15; wrapped axis)

Pre/post periods (aligned with 2017 minimum framing):
  pre  = 1980–2017
  post = 2018–2023
"""

import sys
from pathlib import Path

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
ANOM_DIR = PROJECT_ROOT / "results" / "anomalies"
SECTOR_FILE = PROJECT_ROOT / "data" / "canonical_sectors.nc"

REMOTE_ROOT = "gdrive:sea-ice-phase/Results/Ch2_Figures"
SUBFOLDER = "trends"

# Pre/post definition
PRE_START, PRE_END = 1980, 2017
POST_START, POST_END = 2018, 2023

# Canonical sectors: numeric IDs for logic, labels for plotting
sector_ids = [1, 2, 3, 4, 5]
sector_labels = {
    1: "A–B",   # Amundsen–Bellingshausen
    2: "WED",   # Weddell
    3: "KHV",   # King Haakon VII
    4: "EA",    # East Antarctica
    5: "RA",    # Ross–Amundsen (rename here if your mask is Ross only)
}

# ---------------------------------------------------------------------
# Load anomalies + mask
# ---------------------------------------------------------------------
def load_fs_ms_clim_anom():
    """
    Load FS/MS climatology + anomaly fields for static + dynamic,
    and the sector / ocean mask.

    Notes:
    - MS climatology/anomalies are stored on a wrapped axis (days since Aug 15).
    - Some files may include non-standard 'units' that look like time; we force decode_times=False.
    """
    # FS (normal)
    fs_dyn_clim = xr.open_dataset(ANOM_DIR / "FS_dynamic_climatology.nc")["FS_dynamic_clim"]
    fs_dyn_anom = xr.open_dataset(ANOM_DIR / "FS_dynamic_anomalies.nc")["FS_dynamic_anom"]

    fs_sta_clim = xr.open_dataset(ANOM_DIR / "FS_static_climatology.nc")["FS_static_clim"]
    fs_sta_anom = xr.open_dataset(ANOM_DIR / "FS_static_anomalies.nc")["FS_static_anom"]

    # MS (wrapped days-since axis; avoid any time decoding)
    ms_dyn_ds = xr.open_dataset(ANOM_DIR / "MS_dynamic_climatology.nc", decode_times=False)
    ms_dyn_clim = ms_dyn_ds[list(ms_dyn_ds.data_vars)[0]]
    ms_dyn_ds.close()

    ms_dyn_ds2 = xr.open_dataset(ANOM_DIR / "MS_dynamic_anomalies.nc", decode_times=False)
    ms_dyn_anom = ms_dyn_ds2[list(ms_dyn_ds2.data_vars)[0]]
    ms_dyn_ds2.close()

    ms_sta_ds = xr.open_dataset(ANOM_DIR / "MS_static_climatology.nc", decode_times=False)
    ms_sta_clim = ms_sta_ds[list(ms_sta_ds.data_vars)[0]]
    ms_sta_ds.close()

    ms_sta_ds2 = xr.open_dataset(ANOM_DIR / "MS_static_anomalies.nc", decode_times=False)
    ms_sta_anom = ms_sta_ds2[list(ms_sta_ds2.data_vars)[0]]
    ms_sta_ds2.close()

    ds_mask = xr.open_dataset(SECTOR_FILE)
    valid_ocean = ds_mask["valid_ocean"].astype(bool)
    sector_mask = ds_mask["sector_id"]
    ds_mask.close()

    return {
        "FS_dynamic_clim": fs_dyn_clim,
        "FS_dynamic_anom": fs_dyn_anom,
        "MS_dynamic_clim": ms_dyn_clim,
        "MS_dynamic_anom": ms_dyn_anom,
        "FS_static_clim": fs_sta_clim,
        "FS_static_anom": fs_sta_anom,
        "MS_static_clim": ms_sta_clim,
        "MS_static_anom": ms_sta_anom,
        "valid_ocean": valid_ocean,
        "sector_mask": sector_mask,
    }


# ---------------------------------------------------------------------
# Pre/post mean differences
# ---------------------------------------------------------------------
def compute_pre_post(clim, anom, pre_start, pre_end, post_start, post_end):
    """
    Given climatology (clim[y,x]) and anomalies anom[year,y,x] defined as:
      anom(year) = field(year) − clim

    Reconstruct mean fields over pre and post periods:

      pre_mean  = clim + mean(anom over pre years)
      post_mean = clim + mean(anom over post years)

    return (pre_mean, post_mean, post_minus_pre).
    """
    anom_pre = anom.sel(year=slice(pre_start, pre_end))
    anom_post = anom.sel(year=slice(post_start, post_end))

    pre_mean = clim + anom_pre.mean("year", skipna=True)
    post_mean = clim + anom_post.mean("year", skipna=True)
    diff = post_mean - pre_mean
    return pre_mean, post_mean, diff


def make_sign_class_map(diff_dyn, diff_sta, valid_ocean, thresh=0.0):
    """
    Integer classification map for post−pre diffs.

      0 = background (masked / mixed / near-zero)
      1 = both earlier        (dyn < -thresh and sta < -thresh)
      2 = only dynamic earlier(dyn < -thresh and sta >= -thresh)
      3 = only static earlier (sta < -thresh and dyn >= -thresh)
      4 = both later          (dyn > +thresh and sta > +thresh)

    NOTE: Anything not fitting these bins stays as 0 (grey).
    """
    dyn = diff_dyn.where(valid_ocean).values
    sta = diff_sta.where(valid_ocean).values

    cls = np.zeros_like(dyn, dtype=np.int8)

    both_earlier = (dyn < -thresh) & (sta < -thresh)
    only_dyn = (dyn < -thresh) & ~(sta < -thresh)
    only_sta = (sta < -thresh) & ~(dyn < -thresh)
    both_later = (dyn > +thresh) & (sta > +thresh)

    cls[both_earlier] = 1
    cls[only_dyn] = 2
    cls[only_sta] = 3
    cls[both_later] = 4

    # keep everything else as 0 (grey background)
    cls[~valid_ocean.values] = 0

    return xr.DataArray(cls, coords=diff_dyn.coords, dims=diff_dyn.dims, name="post_pre_class")


def sector_mean_deltas(diff_fs_dyn, diff_fs_sta, diff_ms_dyn, diff_ms_sta, sector_mask, valid_ocean):
    """
    Sector-mean post−pre deltas for dynamic and static.
    Returns tidy DataFrame: phase, sector_id, sector_label, method, delta
    """
    records = []

    def _add(phase, d_dyn, d_sta):
        for sec in sector_ids:
            mask = (sector_mask == sec) & valid_ocean

            dyn_vals = d_dyn.where(mask).values
            sta_vals = d_sta.where(mask).values

            records.append({
                "phase": phase,
                "sector_id": sec,
                "sector_label": sector_labels[sec],
                "method": "Dynamic",
                "delta": float(np.nanmean(dyn_vals)),
            })
            records.append({
                "phase": phase,
                "sector_id": sec,
                "sector_label": sector_labels[sec],
                "method": "Static",
                "delta": float(np.nanmean(sta_vals)),
            })

    _add("FS", diff_fs_dyn, diff_fs_sta)
    _add("MS", diff_ms_dyn, diff_ms_sta)

    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------
# Map plotting helpers
# ---------------------------------------------------------------------
def make_polar_ax(fig, gs, row, col):
    proj = ccrs.SouthPolarStereo()
    ax = fig.add_subplot(gs[row, col], projection=proj)
    ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="0.8", edgecolor="none", zorder=2)
    ax.set_facecolor("white")
    ax.gridlines(draw_labels=False, linewidth=0.3, color="0.7", alpha=0.4, linestyle="--")
    return ax


def plot_sign_class_map(ax, da_class, title):
    """
    Plot categorical sign/agreement class map.
    0 is a grey background (not emphasized in legend).
    """
    cmap = mcolors.ListedColormap(
        [
            "#f0f0f0",  # 0 = background (masked / mixed / near-zero)
            "#2b8cbe",  # 1 = both earlier
            "#41ab5d",  # 2 = only dynamic earlier
            "#fdb462",  # 3 = only static earlier
            "#d73027",  # 4 = both later
        ]
    )
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    im = ax.pcolormesh(
        da_class["x"],
        da_class["y"],
        da_class,
        transform=ccrs.SouthPolarStereo(),
        cmap=cmap,
        norm=norm,
        shading="auto",
    )
    ax.set_title(title, fontsize=9, fontweight="bold")
    return im, bounds


def plot_continuous_diff_map(ax, da, title, vlim=20.0):
    """
    Plot continuous difference map (days), with symmetric limits.
    """
    da2 = da.copy()
    # symmetric limits; keep fixed for comparability
    vmin, vmax = -float(vlim), float(vlim)

    im = ax.pcolormesh(
        da2["x"],
        da2["y"],
        da2,
        transform=ccrs.SouthPolarStereo(),
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        shading="auto",
    )
    ax.set_title(title, fontsize=9, fontweight="bold")
    return im


# ---------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------
def make_combined_figure(fields):
    valid_ocean = fields["valid_ocean"]
    sector_mask = fields["sector_mask"]

    # ---------- Pre/post diffs ----------
    _, _, fs_dyn_diff = compute_pre_post(
        fields["FS_dynamic_clim"], fields["FS_dynamic_anom"],
        PRE_START, PRE_END, POST_START, POST_END
    )
    _, _, fs_sta_diff = compute_pre_post(
        fields["FS_static_clim"], fields["FS_static_anom"],
        PRE_START, PRE_END, POST_START, POST_END
    )

    _, _, ms_dyn_diff = compute_pre_post(
        fields["MS_dynamic_clim"], fields["MS_dynamic_anom"],
        PRE_START, PRE_END, POST_START, POST_END
    )
    _, _, ms_sta_diff = compute_pre_post(
        fields["MS_static_clim"], fields["MS_static_anom"],
        PRE_START, PRE_END, POST_START, POST_END
    )

    # ---------- Level 1 class maps ----------
    # thresh=0 means strict sign; you can set thresh=1.0 to ignore tiny shifts
    fs_class = make_sign_class_map(fs_dyn_diff, fs_sta_diff, valid_ocean, thresh=0.0)
    ms_class = make_sign_class_map(ms_dyn_diff, ms_sta_diff, valid_ocean, thresh=0.0)

    # ---------- Level 2 sector means ----------
    df_sector = sector_mean_deltas(fs_dyn_diff, fs_sta_diff, ms_dyn_diff, ms_sta_diff, sector_mask, valid_ocean)

    # ---------- NEW Level 3 (Option 1): magnitude difference maps ----------
    # (post−pre)_dynamic − (post−pre)_static
    fs_ddiff = (fs_dyn_diff - fs_sta_diff).where(valid_ocean)
    ms_ddiff = (ms_dyn_diff - ms_sta_diff).where(valid_ocean)

    # ---------- Figure layout ----------
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1.3, 1.0, 1.3])

    # Titles: concise but explicit
    title_a = f"(a) FS step change sign ({POST_START}–{POST_END} minus {PRE_START}–{PRE_END})"
    title_b = "(b) FS sector mean step change (days)"
    title_c = "(c) FS: (step change) dynamic − static (days)"

    title_d = f"(d) MS step change sign ({POST_START}–{POST_END} minus {PRE_START}–{PRE_END})"
    title_e = "(e) MS sector mean step change (days)"
    title_f = "(f) MS: (step change) dynamic − static (days)"

    # ----- Row 1: FS -----
    ax_fs_L1 = make_polar_ax(fig, gs, 0, 0)
    im_fs_class, bounds_fs_class = plot_sign_class_map(ax_fs_L1, fs_class, title_a)

    ax_fs_L2 = fig.add_subplot(gs[0, 1])
    df_fs = df_sector[df_sector["phase"] == "FS"].copy()

    methods = ["Static", "Dynamic"]
    colors = {"Static": "#4575b4", "Dynamic": "#d73027"}
    x_positions = np.arange(len(sector_ids))
    width = 0.36

    for i, method in enumerate(methods):
        data = df_fs[df_fs["method"] == method].sort_values("sector_id")
        ax_fs_L2.bar(
            x_positions + (i - 0.5) * width,
            data["delta"].values,
            width=width,
            label=method,
            color=colors[method],
        )

    ax_fs_L2.axhline(0, color="0.4", linewidth=0.8)
    ax_fs_L2.set_xticks(x_positions)
    ax_fs_L2.set_xticklabels([sector_labels[s] for s in sector_ids], rotation=0)
    ax_fs_L2.set_ylabel("ΔFS (days)")
    ax_fs_L2.set_title(title_b, fontweight="bold", fontsize=9)
    ax_fs_L2.legend(frameon=True, fontsize=8)

    ax_fs_L3 = make_polar_ax(fig, gs, 0, 2)
    im_fs_ddiff = plot_continuous_diff_map(ax_fs_L3, fs_ddiff, title_c, vlim=20.0)

    # ----- Row 2: MS -----
    ax_ms_L1 = make_polar_ax(fig, gs, 1, 0)
    im_ms_class, bounds_ms_class = plot_sign_class_map(ax_ms_L1, ms_class, title_d)

    ax_ms_L2 = fig.add_subplot(gs[1, 1])
    df_ms = df_sector[df_sector["phase"] == "MS"].copy()

    for i, method in enumerate(methods):
        data = df_ms[df_ms["method"] == method].sort_values("sector_id")
        ax_ms_L2.bar(
            x_positions + (i - 0.5) * width,
            data["delta"].values,
            width=width,
            label=method if i == 0 else None,  # show only once
            color=colors[method],
        )

    ax_ms_L2.axhline(0, color="0.4", linewidth=0.8)
    ax_ms_L2.set_xticks(x_positions)
    ax_ms_L2.set_xticklabels([sector_labels[s] for s in sector_ids], rotation=0)
    ax_ms_L2.set_ylabel("ΔMS (days)")
    ax_ms_L2.set_title(title_e, fontweight="bold", fontsize=9)

    ax_ms_L3 = make_polar_ax(fig, gs, 1, 2)
    im_ms_ddiff = plot_continuous_diff_map(ax_ms_L3, ms_ddiff, title_f, vlim=20.0)

    # ---------- Colorbars ----------
    # (1) Class colorbar shared (FS/MS)
    cax1 = fig.add_axes([0.08, 0.06, 0.35, 0.02])
    cb1 = fig.colorbar(
        im_fs_class,
        cax=cax1,
        orientation="horizontal",
        boundaries=bounds_fs_class,
        ticks=[1, 2, 3, 4],  # IMPORTANT: do not label 0 background
    )
    cb1.set_label("Step-change sign agreement class", fontsize=9)
    cb1.ax.set_xticklabels(
        ["both earlier", "only dynamic earlier", "only static earlier", "both later"],
        fontsize=7,
    )
    cb1.outline.set_visible(False)

    # (2) Continuous ddiff colorbar shared (FS/MS)
    cax2 = fig.add_axes([0.57, 0.06, 0.35, 0.02])
    cb2 = fig.colorbar(im_fs_ddiff, cax=cax2, orientation="horizontal")
    cb2.set_label("(post−pre) dynamic − static (days)", fontsize=9)
    cb2.outline.set_visible(False)

    # Avoid tight_layout warning (manual axes used for colorbars)
    fig.subplots_adjust(left=0.04, right=0.98, top=0.96, bottom=0.12, wspace=0.25, hspace=0.18)

    # ---------- Save / upload ----------
    fig_name = (
        f"Fig_FS_MS_stepchange_comparison_static_vs_dynamic_"
        f"pre{PRE_START}-{PRE_END}_post{POST_START}-{POST_END}.png"
    )
    out_path = get_fig_path(PROJECT_ROOT, subfolder=SUBFOLDER, fig_name=fig_name)

    save_and_upload(
        fig,
        out_path,
        remote_root=REMOTE_ROOT,
        remote_subdir=SUBFOLDER,
    )


def main():
    fields = load_fs_ms_clim_anom()
    make_combined_figure(fields)


if __name__ == "__main__":
    main()

