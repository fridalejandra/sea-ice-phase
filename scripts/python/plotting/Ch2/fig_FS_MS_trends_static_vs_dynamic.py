#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FS/MS trend comparison: static vs dynamic (all 3 levels in one figure).

- Level 1 (col 1): pre–post sign-classification map
- Level 2 (col 2): sector-mean ΔFS / ΔMS barplots
- Level 3 (col 3): trend-agreement map from full anomalies

Rows:
  Row 1: FS
  Row 2: MS

Pre/post periods:
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

# Canonical sectors
sector_ids = [1, 2, 3, 4, 5]
sector_labels = {
    1: "A–B",   # Amundsen–Bellingshausen
    2: "WED",   # Weddell
    3: "KHV",   # King Haakon VII
    4: "EA",    # East Antarctica
    5: "RA",    # Ross–Amundsen
}

# ---------------------------------------------------------------------
# Load anomalies + mask
# ---------------------------------------------------------------------
def load_fs_ms_clim_anom():
    """Load FS/MS clim + anoms for static + dynamic, plus masks."""
    fs_dyn_clim = xr.open_dataset(ANOM_DIR / "FS_dynamic_climatology.nc")[
        "FS_dynamic_clim"
    ]
    fs_dyn_anom = xr.open_dataset(ANOM_DIR / "FS_dynamic_anomalies.nc")[
        "FS_dynamic_anom"
    ]

    ms_dyn_clim = xr.open_dataset(ANOM_DIR / "MS_dynamic_climatology.nc")[
        "MS_dynamic_clim"
    ]
    ms_dyn_anom = xr.open_dataset(ANOM_DIR / "MS_dynamic_anomalies.nc")[
        "MS_dynamic_anom"
    ]

    fs_sta_clim = xr.open_dataset(ANOM_DIR / "FS_static_climatology.nc")[
        "FS_static_clim"
    ]
    fs_sta_anom = xr.open_dataset(ANOM_DIR / "FS_static_anomalies.nc")[
        "FS_static_anom"
    ]

    ms_sta_clim = xr.open_dataset(ANOM_DIR / "MS_static_climatology.nc")[
        "MS_static_clim"
    ]
    ms_sta_anom = xr.open_dataset(ANOM_DIR / "MS_static_anomalies.nc")[
        "MS_static_anom"
    ]

    ds_mask = xr.open_dataset(SECTOR_FILE)
    valid_ocean = ds_mask["valid_ocean"].astype(bool)
    sector_mask = ds_mask["sector_id"]
    ds_mask.close()

    # Phase-specific climatology masks: only keep pixels with non-zero clim in BOTH methods
    fs_clim_mask = valid_ocean & (fs_dyn_clim > 0) & (fs_sta_clim > 0)
    ms_clim_mask = valid_ocean & (ms_dyn_clim > 0) & (ms_sta_clim > 0)

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
        "fs_clim_mask": fs_clim_mask,
        "ms_clim_mask": ms_clim_mask,
    }


# ---------------------------------------------------------------------
# Pre/post mean differences (LEVEL 1 & 2)
# ---------------------------------------------------------------------
def compute_pre_post(clim, anom, pre_start, pre_end, post_start, post_end):
    """Reconstruct pre/post means from clim + anomalies."""
    anom_pre = anom.sel(year=slice(pre_start, pre_end))
    anom_post = anom.sel(year=slice(post_start, post_end))

    pre_mean = clim + anom_pre.mean("year", skipna=True)
    post_mean = clim + anom_post.mean("year", skipna=True)
    diff = post_mean - pre_mean
    return pre_mean, post_mean, diff


def make_sign_class_map(diff_dyn, diff_sta, valid_ocean, thresh=0.0):
    """
    Integer classification:

      0 = masked/other
      1 = both earlier
      2 = only dynamic earlier
      3 = only static earlier
      4 = both later
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
    cls[~valid_ocean.values] = 0

    return xr.DataArray(
        cls,
        coords=diff_dyn.coords,
        dims=diff_dyn.dims,
        name="pre_post_sign_class",
    )


def sector_mean_deltas(diff_dyn, diff_sta, sector_mask, fs_mask, ms_mask):
    """
    Sector-mean Δ (post−pre) for dynamic and static, using phase-specific
    climatology masks so that only pixels with non-zero clim are included.
    """
    records = []

    for phase_name, d_dyn, d_sta, phase_mask in [
        ("FS", diff_dyn["FS"], diff_sta["FS"], fs_mask),
        ("MS", diff_dyn["MS"], diff_sta["MS"], ms_mask),
    ]:
        for sec in sector_ids:
            mask = (sector_mask == sec) & phase_mask

            dyn_vals = d_dyn.where(mask).values
            sta_vals = d_sta.where(mask).values

            dyn_mean = float(np.nanmean(dyn_vals))
            sta_mean = float(np.nanmean(sta_vals))

            records.append(
                {
                    "phase": phase_name,
                    "sector_id": sec,
                    "sector_label": sector_labels[sec],
                    "method": "Dynamic",
                    "delta": dyn_mean,
                }
            )
            records.append(
                {
                    "phase": phase_name,
                    "sector_id": sec,
                    "sector_label": sector_labels[sec],
                    "method": "Static",
                    "delta": sta_mean,
                }
            )

    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------
# Linear trends from anomalies (LEVEL 3)
# ---------------------------------------------------------------------
def _slope_from_anom(y, years):
    mask = np.isfinite(y)
    if mask.sum() < 2:
        return np.nan
    return float(np.polyfit(years[mask], y[mask], 1)[0])


def compute_trend_slopes(anom_da):
    years = anom_da["year"].values.astype(float)
    slopes = xr.apply_ufunc(
        _slope_from_anom,
        anom_da,
        kwargs={"years": years},
        input_core_dims=[["year"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
    slopes.name = "slope"
    return slopes


def make_trend_agreement_map(slope_dyn, slope_sta, valid_ocean):
    dyn = slope_dyn.where(valid_ocean).values
    sta = slope_sta.where(valid_ocean).values

    agree = (dyn < 0.0) & (sta < 0.0)
    arr = np.zeros_like(dyn, dtype=np.int8)
    arr[agree] = 1
    arr[~valid_ocean.values] = 0

    return xr.DataArray(
        arr,
        coords=slope_dyn.coords,
        dims=slope_dyn.dims,
        name="trend_earlier_both",
    )


# ---------------------------------------------------------------------
# Map plotting helpers
# ---------------------------------------------------------------------
def make_polar_ax(fig, gs, row, col):
    proj = ccrs.SouthPolarStereo()
    ax = fig.add_subplot(gs[row, col], projection=proj)
    ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="0.8", edgecolor="none", zorder=2)
    ax.set_facecolor("white")
    ax.gridlines(
        draw_labels=False,
        linewidth=0.3,
        color="0.7",
        alpha=0.4,
        linestyle="--",
    )
    return ax


def plot_sign_class_map(ax, da_class, title):
    """Pre–post classification map."""
    cmap = mcolors.ListedColormap(
        [
            "#f0f0f0",  # 0 = mask/land
            "#2b8cbe",  # 1 = both earlier
            "#41ab5d",  # 2 = only dynamic
            "#fdb462",  # 3 = only static
            "#d73027",  # 4 = both later
        ]
    )
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    x = da_class["x"]
    y = da_class["y"]

    im = ax.pcolormesh(
        x,
        y,
        da_class,
        transform=ccrs.SouthPolarStereo(),
        cmap=cmap,
        norm=norm,
        shading="auto",
    )
    ax.set_title(title, fontsize=9, fontweight="bold")
    return im, bounds, cmap


def plot_trend_agreement_map(ax, da_agree, title):
    """Trend-agreement map: 1 = both earlier, 0 = other."""
    cmap = mcolors.ListedColormap(["#f0f0f0", "#2b8cbe"])
    bounds = [-0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    x = da_agree["x"]
    y = da_agree["y"]

    im = ax.pcolormesh(
        x,
        y,
        da_agree,
        transform=ccrs.SouthPolarStereo(),
        cmap=cmap,
        norm=norm,
        shading="auto",
    )
    ax.set_title(title, fontsize=9, fontweight="bold")
    return im, bounds, cmap


# ---------------------------------------------------------------------
# Main plotting: 2x3 figure
# ---------------------------------------------------------------------
def make_combined_figure(fields):
    valid_ocean = fields["valid_ocean"]
    sector_mask = fields["sector_mask"]
    fs_mask = fields["fs_clim_mask"]
    ms_mask = fields["ms_clim_mask"]

    # ---------- Level 1: pre/post diffs ----------
    _, _, fs_dyn_diff = compute_pre_post(
        fields["FS_dynamic_clim"],
        fields["FS_dynamic_anom"],
        PRE_START,
        PRE_END,
        POST_START,
        POST_END,
    )
    _, _, fs_sta_diff = compute_pre_post(
        fields["FS_static_clim"],
        fields["FS_static_anom"],
        PRE_START,
        PRE_END,
        POST_START,
        POST_END,
    )
    _, _, ms_dyn_diff = compute_pre_post(
        fields["MS_dynamic_clim"],
        fields["MS_dynamic_anom"],
        PRE_START,
        PRE_END,
        POST_START,
        POST_END,
    )
    _, _, ms_sta_diff = compute_pre_post(
        fields["MS_static_clim"],
        fields["MS_static_anom"],
        PRE_START,
        PRE_END,
        POST_START,
        POST_END,
    )

    diff_dyn = xr.Dataset({"FS": fs_dyn_diff, "MS": ms_dyn_diff})
    diff_sta = xr.Dataset({"FS": fs_sta_diff, "MS": ms_sta_diff})

    fs_class = make_sign_class_map(fs_dyn_diff, fs_sta_diff, valid_ocean)
    ms_class = make_sign_class_map(ms_dyn_diff, ms_sta_diff, valid_ocean)

    # ---------- Level 2: sector mean deltas (using clim masks) ----------
    df_sector = sector_mean_deltas(
        diff_dyn, diff_sta, sector_mask, fs_mask, ms_mask
    )

    # ---------- Level 3: trend agreement ----------
    years = fields["FS_dynamic_anom"]["year"].values
    print(f"[INFO] Trend years span: {years.min()}–{years.max()}")

    fs_slope_dyn = compute_trend_slopes(fields["FS_dynamic_anom"])
    fs_slope_sta = compute_trend_slopes(fields["FS_static_anom"])
    fs_agree = make_trend_agreement_map(fs_slope_dyn, fs_slope_sta, valid_ocean)

    ms_slope_dyn = compute_trend_slopes(fields["MS_dynamic_anom"])
    ms_slope_sta = compute_trend_slopes(fields["MS_static_anom"])
    ms_agree = make_trend_agreement_map(ms_slope_dyn, ms_slope_sta, valid_ocean)

    # ---------- Fractions for text ----------
    ocean_n = valid_ocean.values.sum()

    def frac_both_earlier(class_da):
        arr = class_da.values
        return float((arr == 1).sum()) / ocean_n

    def frac_trend_agree(agree_da):
        arr = agree_da.values
        return float((arr == 1).sum()) / ocean_n

    print("[INFO] Pre–post both earlier (FS):", frac_both_earlier(fs_class))
    print("[INFO] Pre–post both earlier (MS):", frac_both_earlier(ms_class))
    print("[INFO] Trend both earlier (FS):", frac_trend_agree(fs_agree))
    print("[INFO] Trend both earlier (MS):", frac_trend_agree(ms_agree))

    # ---------- Figure layout ----------
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(
        2, 3, height_ratios=[1, 1], width_ratios=[1.3, 1.0, 1.3]
    )

    # ----- Row 1: FS -----
    ax_fs_L1 = make_polar_ax(fig, gs, 0, 0)
    im_fs_class, bounds_fs_class, _ = plot_sign_class_map(
        ax_fs_L1, fs_class, "(a) FS pre–post sign (static vs dynamic)"
    )

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

    ax_fs_L2.axhline(0, color="0.2", linewidth=0.8)
    ax_fs_L2.set_xticks(x_positions)
    ax_fs_L2.set_xticklabels(
        [sector_labels[s] for s in sector_ids], rotation=0
    )
    ax_fs_L2.set_ylabel("Sector mean ΔFS (days)")
    ax_fs_L2.set_title("(b) FS sector mean Δ", fontweight="bold", fontsize=9)
    # symmetric y-limits around zero
    fs_max = float(np.nanmax(np.abs(df_fs["delta"].values)))
    ax_fs_L2.set_ylim(-fs_max * 1.1, fs_max * 1.1)
    ax_fs_L2.legend(frameon=True, fontsize=8)

    ax_fs_L3 = make_polar_ax(fig, gs, 0, 2)
    im_fs_trend, bounds_fs_trend, _ = plot_trend_agreement_map(
        ax_fs_L3, fs_agree, "(c) FS trend agreement (both earlier)"
    )

    # ----- Row 2: MS -----
    ax_ms_L1 = make_polar_ax(fig, gs, 1, 0)
    im_ms_class, bounds_ms_class, _ = plot_sign_class_map(
        ax_ms_L1, ms_class, "(d) MS pre–post sign (static vs dynamic)"
    )

    ax_ms_L2 = fig.add_subplot(gs[1, 1])
    df_ms = df_sector[df_sector["phase"] == "MS"].copy()

    for i, method in enumerate(methods):
        data = df_ms[df_ms["method"] == method].sort_values("sector_id")
        ax_ms_L2.bar(
            x_positions + (i - 0.5) * width,
            data["delta"].values,
            width=width,
            label=method if i == 0 else None,
            color=colors[method],
        )

    ax_ms_L2.axhline(0, color="0.2", linewidth=0.8)
    ax_ms_L2.set_xticks(x_positions)
    ax_ms_L2.set_xticklabels(
        [sector_labels[s] for s in sector_ids], rotation=0
    )
    ax_ms_L2.set_ylabel("Sector mean ΔMS (days)")
    ax_ms_L2.set_title("(e) MS sector mean Δ", fontweight="bold", fontsize=9)
    ms_max = float(np.nanmax(np.abs(df_ms["delta"].values)))
    ax_ms_L2.set_ylim(-ms_max * 1.1, ms_max * 1.1)

    ax_ms_L3 = make_polar_ax(fig, gs, 1, 2)
    im_ms_trend, bounds_ms_trend, _ = plot_trend_agreement_map(
        ax_ms_L3, ms_agree, "(f) MS trend agreement (both earlier)"
    )

    # ---------- Colorbars ----------
    cax1 = fig.add_axes([0.08, 0.06, 0.35, 0.02])
    cb1 = fig.colorbar(
        im_fs_class,
        cax=cax1,
        orientation="horizontal",
        boundaries=bounds_fs_class,
        ticks=[0, 1, 2, 3, 4],
    )
    cb1.set_label("Pre–post class", fontsize=9)
    cb1.ax.set_xticklabels(
        ["mask/land", "both earlier", "only dyn", "only stat", "both later"],
        fontsize=7,
    )
    cb1.outline.set_visible(False)

    cax3 = fig.add_axes([0.57, 0.06, 0.35, 0.02])
    cb3 = fig.colorbar(
        im_fs_trend,
        cax=cax3,
        orientation="horizontal",
        boundaries=bounds_fs_trend,
        ticks=[0, 1],
    )
    cb3.set_label("Trend class", fontsize=9)
    cb3.ax.set_xticklabels(["other", "both earlier"], fontsize=7)
    cb3.outline.set_visible(False)

    fig.tight_layout(rect=[0.02, 0.10, 0.98, 0.98])

    fig_name = (
        f"Fig_FS_MS_trend_comparison_static_vs_dynamic_"
        f"pre{PRE_START}-{PRE_END}_post{POST_START}-{POST_END}.png"
    )
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
    fields = load_fs_ms_clim_anom()
    make_combined_figure(fields)


if __name__ == "__main__":
    main()
