#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FS/MS volatility comparison: static vs dynamic.

We quantify changes in interannual variability (volatility) of freeze start (FS)
and melt start (MS) between:

  pre  = 1980–2017
  post = 2018–2023

For each gridcell and method (static / dynamic), we compute:

    σ_pre  = std(phase_anom[pre years])
    σ_post = std(phase_anom[post years])
    Δσ     = σ_post − σ_pre   [days]

Positive Δσ = more volatile in the post-2018 regime.
Negative Δσ = more stable.

Figure layout:

Row 1 (FS):
  (a) Δσ_FS dynamic (map)
  (b) Δσ_FS static  (map)
  (c) Sector-mean Δσ_FS (bars: static & dynamic)

Row 2 (MS):
  (d) Δσ_MS dynamic (map)
  (e) Δσ_MS static  (map)
  (f) Sector-mean Δσ_MS (bars)
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
SUBFOLDER   = "volatility"

# Pre/post definition (mirror the trend figure)
PRE_START, PRE_END   = 1980, 2017
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

# volatility color range (days)
VMAX_VOL = 20.0  # tweak if needed


# ---------------------------------------------------------------------
# Load anomalies + masks
# ---------------------------------------------------------------------
def load_fs_ms_anoms_and_masks():
    """
    Load FS/MS anomaly fields for static + dynamic and the sector/ocean masks.
    We don't need climatologies here except to build phase-specific valid masks.
    """
    fs_dyn_anom = xr.open_dataset(ANOM_DIR / "FS_dynamic_anomalies.nc")[
        "FS_dynamic_anom"
    ]
    ms_dyn_anom = xr.open_dataset(ANOM_DIR / "MS_dynamic_anomalies.nc")[
        "MS_dynamic_anom"
    ]

    fs_sta_anom = xr.open_dataset(ANOM_DIR / "FS_static_anomalies.nc")[
        "FS_static_anom"
    ]
    ms_sta_anom = xr.open_dataset(ANOM_DIR / "MS_static_anomalies.nc")[
        "MS_static_anom"
    ]

    # We can use the climatologies just to identify "valid" phase pixels
    fs_dyn_clim = xr.open_dataset(ANOM_DIR / "FS_dynamic_climatology.nc")[
        "FS_dynamic_clim"
    ]
    fs_sta_clim = xr.open_dataset(ANOM_DIR / "FS_static_climatology.nc")[
        "FS_static_clim"
    ]
    ms_dyn_clim = xr.open_dataset(ANOM_DIR / "MS_dynamic_climatology.nc")[
        "MS_dynamic_clim"
    ]
    ms_sta_clim = xr.open_dataset(ANOM_DIR / "MS_static_climatology.nc")[
        "MS_static_clim"
    ]

    ds_mask = xr.open_dataset(SECTOR_FILE)
    valid_ocean = ds_mask["valid_ocean"].astype(bool)
    sector_mask = ds_mask["sector_id"]
    ds_mask.close()

    # Require non-zero climatology in BOTH methods for volatility stats
    fs_valid = valid_ocean & (fs_dyn_clim > 0) & (fs_sta_clim > 0)
    ms_valid = valid_ocean & (ms_dyn_clim > 0) & (ms_sta_clim > 0)

    return {
        "FS_dynamic_anom": fs_dyn_anom,
        "MS_dynamic_anom": ms_dyn_anom,
        "FS_static_anom":  fs_sta_anom,
        "MS_static_anom":  ms_sta_anom,
        "valid_ocean":     valid_ocean,
        "sector_mask":     sector_mask,
        "fs_valid":        fs_valid,
        "ms_valid":        ms_valid,
    }


# ---------------------------------------------------------------------
# Volatility computation
# ---------------------------------------------------------------------
def compute_volatility_delta(anom_da, pre_start, pre_end, post_start, post_end):
    """
    Given anomalies anom_da[year, y, x], compute interannual std dev
    in pre and post periods and their difference Δσ = σ_post − σ_pre.

    Returns (sigma_pre, sigma_post, delta_sigma) [all days].
    """
    anom_pre = anom_da.sel(year=slice(pre_start, pre_end))
    anom_post = anom_da.sel(year=slice(post_start, post_end))

    sigma_pre = anom_pre.std("year", skipna=True)
    sigma_post = anom_post.std("year", skipna=True)

    delta = sigma_post - sigma_pre
    return sigma_pre, sigma_post, delta


def sector_mean_vol_deltas(delta_dyn, delta_sta, sector_mask, valid_mask):
    """
    Compute sector-mean Δσ for dynamic and static.

    delta_*: [y, x] arrays (FS or MS)
    valid_mask: phase-specific valid mask (fs_valid or ms_valid)

    Returns tidy DataFrame with:
      phase, sector_id, sector_label, method, delta_sigma
    """
    records = []

    for phase_name, d_dyn, d_sta, phase_valid in [
        ("FS", delta_dyn["FS"], delta_sta["FS"], valid_mask["fs_valid"]),
        ("MS", delta_dyn["MS"], delta_sta["MS"], valid_mask["ms_valid"]),
    ]:
        for sec in sector_ids:
            mask = (sector_mask == sec) & phase_valid

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
                    "delta_sigma": dyn_mean,
                }
            )
            records.append(
                {
                    "phase": phase_name,
                    "sector_id": sec,
                    "sector_label": sector_labels[sec],
                    "method": "Static",
                    "delta_sigma": sta_mean,
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
        linewidth=0.3,
        color="0.7",
        alpha=0.4,
        linestyle="--",
    )
    return ax


def plot_vol_map(ax, da, title, vmax=VMAX_VOL):
    """
    Plot Δσ volatility map (days). Positive = more volatile post-2018.
    """
    cmap = plt.get_cmap("RdBu_r")
    x = da["x"]
    y = da["y"]

    im = ax.pcolormesh(
        x,
        y,
        da,
        transform=ccrs.SouthPolarStereo(),
        cmap=cmap,
        vmin=-vmax,
        vmax=+vmax,
        shading="auto",
    )
    ax.set_title(title, fontsize=9, fontweight="bold")
    return im


# ---------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------
def make_volatility_figure(fields):
    valid_ocean = fields["valid_ocean"]
    sector_mask = fields["sector_mask"]
    fs_valid = fields["fs_valid"]
    ms_valid = fields["ms_valid"]

    # ---------- Compute volatility Δσ for FS / MS, dynamic / static ----------
    fs_dyn_anom = fields["FS_dynamic_anom"]
    fs_sta_anom = fields["FS_static_anom"]
    ms_dyn_anom = fields["MS_dynamic_anom"]
    ms_sta_anom = fields["MS_static_anom"]

    years = fs_dyn_anom["year"].values
    print(f"[INFO] Volatility years span: {years.min()}–{years.max()}")

    # FS
    _, _, fs_dyn_delta = compute_volatility_delta(
        fs_dyn_anom, PRE_START, PRE_END, POST_START, POST_END
    )
    _, _, fs_sta_delta = compute_volatility_delta(
        fs_sta_anom, PRE_START, PRE_END, POST_START, POST_END
    )

    # MS
    _, _, ms_dyn_delta = compute_volatility_delta(
        ms_dyn_anom, PRE_START, PRE_END, POST_START, POST_END
    )
    _, _, ms_sta_delta = compute_volatility_delta(
        ms_sta_anom, PRE_START, PRE_END, POST_START, POST_END
    )

    # Mask by valid ocean + phase-specific valid masks
    fs_dyn_delta = fs_dyn_delta.where(fs_valid & valid_ocean)
    fs_sta_delta = fs_sta_delta.where(fs_valid & valid_ocean)
    ms_dyn_delta = ms_dyn_delta.where(ms_valid & valid_ocean)
    ms_sta_delta = ms_sta_delta.where(ms_valid & valid_ocean)

    # Package into Datasets for sector stats
    delta_dyn = xr.Dataset({"FS": fs_dyn_delta, "MS": ms_dyn_delta})
    delta_sta = xr.Dataset({"FS": fs_sta_delta, "MS": ms_sta_delta})
    valid_dict = {"fs_valid": fs_valid, "ms_valid": ms_valid}

    # ---------- Sector mean volatility deltas ----------
    df_sector = sector_mean_vol_deltas(
        delta_dyn, delta_sta, sector_mask, valid_dict
    )

    # ---------- Figure layout ----------
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(
        2, 3, height_ratios=[1, 1], width_ratios=[1.3, 1.3, 1.0]
    )

    # ----- Row 1: FS volatility -----
    ax_fs_dyn = make_polar_ax(fig, gs, 0, 0)
    im_fs_dyn = plot_vol_map(
        ax_fs_dyn,
        fs_dyn_delta,
        "(a) Δσ_FS dynamic (post − pre)",
        vmax=VMAX_VOL,
    )

    ax_fs_sta = make_polar_ax(fig, gs, 0, 1)
    im_fs_sta = plot_vol_map(
        ax_fs_sta,
        fs_sta_delta,
        "(b) Δσ_FS static (post − pre)",
        vmax=VMAX_VOL,
    )

    # FS sector bars
    ax_fs_bar = fig.add_subplot(gs[0, 2])
    df_fs = df_sector[df_sector["phase"] == "FS"].copy()

    methods = ["Static", "Dynamic"]
    colors = {"Static": "#4575b4", "Dynamic": "#d73027"}
    x_positions = np.arange(len(sector_ids))
    width = 0.36

    for i, method in enumerate(methods):
        data = df_fs[df_fs["method"] == method].sort_values("sector_id")
        ax_fs_bar.bar(
            x_positions + (i - 0.5) * width,
            data["delta_sigma"].values,
            width=width,
            label=method,
            color=colors[method],
        )

    ax_fs_bar.axhline(0, color="0.2", linewidth=0.8)
    ax_fs_bar.set_xticks(x_positions)
    ax_fs_bar.set_xticklabels(
        [sector_labels[s] for s in sector_ids], rotation=0
    )
    ax_fs_bar.set_ylabel("Δσ_FS (post − pre, days)")
    ax_fs_bar.set_title("(c) FS sector mean Δσ", fontweight="bold", fontsize=9)
    fs_max = float(np.nanmax(np.abs(df_fs["delta_sigma"].values)))
    if np.isfinite(fs_max) and fs_max > 0:
        ax_fs_bar.set_ylim(-fs_max * 1.2, fs_max * 1.2)
    ax_fs_bar.legend(frameon=True, fontsize=8)

    # ----- Row 2: MS volatility -----
    ax_ms_dyn = make_polar_ax(fig, gs, 1, 0)
    im_ms_dyn = plot_vol_map(
        ax_ms_dyn,
        ms_dyn_delta,
        "(d) Δσ_MS dynamic (post − pre)",
        vmax=VMAX_VOL,
    )

    ax_ms_sta = make_polar_ax(fig, gs, 1, 1)
    im_ms_sta = plot_vol_map(
        ax_ms_sta,
        ms_sta_delta,
        "(e) Δσ_MS static (post − pre)",
        vmax=VMAX_VOL,
    )

    ax_ms_bar = fig.add_subplot(gs[1, 2])
    df_ms = df_sector[df_sector["phase"] == "MS"].copy()

    for i, method in enumerate(methods):
        data = df_ms[df_ms["method"] == method].sort_values("sector_id")
        ax_ms_bar.bar(
            x_positions + (i - 0.5) * width,
            data["delta_sigma"].values,
            width=width,
            label=method if i == 0 else None,
            color=colors[method],
        )

    ax_ms_bar.axhline(0, color="0.2", linewidth=0.8)
    ax_ms_bar.set_xticks(x_positions)
    ax_ms_bar.set_xticklabels(
        [sector_labels[s] for s in sector_ids], rotation=0
    )
    ax_ms_bar.set_ylabel("Δσ_MS (post − pre, days)")
    ax_ms_bar.set_title("(f) MS sector mean Δσ", fontweight="bold", fontsize=9)
    ms_max = float(np.nanmax(np.abs(df_ms["delta_sigma"].values)))
    if np.isfinite(ms_max) and ms_max > 0:
        ax_ms_bar.set_ylim(-ms_max * 1.2, ms_max * 1.2)

    # ---------- Shared colorbar ----------
    cax = fig.add_axes([0.15, 0.06, 0.55, 0.02])
    cb = fig.colorbar(
        im_fs_dyn,
        cax=cax,
        orientation="horizontal",
    )
    cb.set_label("Δσ (post − pre, days)", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_visible(False)

    fig.tight_layout(rect=[0.02, 0.10, 0.98, 0.98])

    # ---------- Save / upload ----------
    fig_name = (
        f"Fig_FS_MS_volatility_static_vs_dynamic_"
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
    fields = load_fs_ms_anoms_and_masks()
    make_volatility_figure(fields)


if __name__ == "__main__":
    main()
