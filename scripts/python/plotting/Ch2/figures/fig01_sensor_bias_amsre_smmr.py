#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig01_sensor_bias_amsre_smmr.py

Compare AMSRE vs SMMR phase dates (static thr15 k5) and produce:
  (a) MS climatological bias map (AMSRE − SMMR, days)
  (b) FS climatological bias map (AMSRE − SMMR, days)
  (c) Stacked histogram of wrapped bias for both phases

Inputs:
  data/SMMR_phase/static/thr15_k5/FS/FS_YYYY.nc  — variable: FS
  data/SMMR_phase/static/thr15_k5/MS/MS_YYYY.nc  — variable: MS
  data/AMSRE_phase/static/thr15_k5/FS/FS_YYYY.nc — variable: FS
  data/AMSRE_phase/static/thr15_k5/MS/MS_YYYY.nc — variable: MS

Output:
  results/Ch2_Figures/Fig01_sensor_advance_retreat_bias_hist_AMSREminusSMMR_2012-2024.png
"""

import sys
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------
# project root
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.python.plotting.ch2_fig_utils import (  # noqa: E402
    set_mpl_defaults,
    format_fig_name,
    get_fig_path,
    save_and_upload,
)

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
YEARS       = range(2012, 2025)   # 2012–2024 inclusive
THR         = 15
K           = 5
SMMR_ROOT   = PROJECT_ROOT / "data" / "SMMR_phase" / "static" / f"thr{THR:02d}_k{K}"
AMSRE_ROOT  = PROJECT_ROOT / "data" / "AMSRE_phase" / "static" / f"thr{THR:02d}_k{K}"
SUBFOLDER   = ""
REMOTE_ROOT = "gdrive:sea-ice-phase/results/Ch2_Figures"


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def wrapped_difference(a, b, period=365.0):
    """Circular difference (a - b) mapped into [-period/2, period/2]."""
    return (a - b + period / 2.0) % period - period / 2.0


def load_phase_year(root: Path, phase: str, year: int) -> xr.DataArray:
    path = root / phase / f"{phase}_{year}.nc"
    ds   = xr.open_dataset(path)
    da   = ds[phase].load()
    ds.close()
    return da


def compute_bias(phase: str) -> tuple[xr.DataArray, np.ndarray]:
    """
    Compute annual and climatological wrapped bias (AMSRE − SMMR) for a phase.
    Returns (bias_clim [y,x], all_bias_flat 1D).
    """
    yearly = []
    flat   = []

    for year in YEARS:
        print(f"  {phase} {year}")
        try:
            smmr  = load_phase_year(SMMR_ROOT, phase, year)
            amsre = load_phase_year(AMSRE_ROOT, phase, year)
        except FileNotFoundError as e:
            print(f"    skipping: {e}")
            continue

        # AMSRE is on a finer grid — coarsen to SMMR resolution
        ny, nx = amsre.shape
        if ny % 2 != 0:
            amsre = amsre.isel(y=slice(0, ny - 1))
        if nx % 2 != 0:
            amsre = amsre.isel(x=slice(0, nx - 1))
        amsre_coarse = amsre.coarsen(y=2, x=2, boundary="trim").mean()
        amsre_coarse = amsre_coarse.assign_coords(x=smmr.x, y=smmr.y)

        bias_vals = wrapped_difference(amsre_coarse.values, smmr.values)
        bias = xr.DataArray(
            data=bias_vals,
            coords={"y": smmr.y, "x": smmr.x},
            dims=("y", "x"),
        )
        yearly.append(bias)
        flat.append(bias_vals.ravel())

    bias_clim = xr.concat(yearly, dim="year").mean("year", skipna=True)
    all_bias  = np.concatenate(flat)
    all_bias  = all_bias[~np.isnan(all_bias)]
    return bias_clim, all_bias


def plot_bias_map(ax, bias_da, vlim=20):
    proj = ccrs.SouthPolarStereo()
    im   = ax.pcolormesh(
        bias_da["x"].values, bias_da["y"].values, bias_da.values,
        transform=proj, cmap="RdBu_r", vmin=-vlim, vmax=vlim, shading="auto",
    )
    ax.set_extent([-180, 180, -90, -50], crs=ccrs.PlateCarree())
    ax.set_facecolor("white")
    ax.coastlines(color="0.4", linewidth=0.5)
    ax.add_feature(
        cfeature.LAND.with_scale("110m"),
        facecolor="0.85", edgecolor="0.6", linewidth=0.4, zorder=3,
    )
    return im


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    set_mpl_defaults()

    print("Computing MS bias...")
    bias_clim_ms, all_bias_ms = compute_bias("MS")
    print("Computing FS bias...")
    bias_clim_fs, all_bias_fs = compute_bias("FS")

    fig  = plt.figure(figsize=(14, 5))
    proj = ccrs.SouthPolarStereo()
    gs   = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.05, 1.2], wspace=0.3)

    ax_ms   = fig.add_subplot(gs[0, 0], projection=proj)
    ax_fs   = fig.add_subplot(gs[0, 1], projection=proj)
    ax_hist = fig.add_subplot(gs[0, 2])

    for ax, label in [(ax_ms, "(a)"), (ax_fs, "(b)"), (ax_hist, "(c)")]:
        ax.text(0.02, 0.98, label, transform=ax.transAxes,
                ha="left", va="top", fontsize=12, fontweight="bold")

    title_years = f"{YEARS.start}–{YEARS.stop - 1}"
    ax_ms.set_title(f"Melt Start ({title_years})", fontsize=9)
    ax_fs.set_title(f"Freeze Start ({title_years})", fontsize=9)

    im_ms = plot_bias_map(ax_ms, bias_clim_ms)
    im_fs = plot_bias_map(ax_fs, bias_clim_fs)

    cbar = fig.colorbar(
        im_ms, ax=[ax_ms, ax_fs],
        orientation="horizontal", pad=0.08, shrink=0.9,
    )
    cbar.set_label("Bias (AMSRE − SMMR, days)")

    # histogram
    bias_all  = np.concatenate([all_bias_fs, all_bias_ms])
    phase_all = (["Advance"] * len(all_bias_fs)) + (["Retreat"] * len(all_bias_ms))
    df = pd.DataFrame({"bias": bias_all, "phase": phase_all})

    sns.histplot(
        data=df, x="bias", hue="phase", multiple="stack",
        bins=np.arange(-40, 42, 2), stat="density", common_norm=False,
        edgecolor=".3", linewidth=0.5, ax=ax_hist,
    )
    ax_hist.set_ylim(0, 0.08)
    ax_hist.axvline(0, color="k", linewidth=0.8)
    ax_hist.set_xlabel("Bias (AMSRE − SMMR, days)")
    ax_hist.set_ylabel("Probability density")
    ax_hist.set_xlim(-40, 40)
    ax_hist.set_xticks([-40, -20, 0, 20, 40])
    leg = ax_hist.get_legend()
    if leg:
        leg.set_title("")
    sns.despine(fig=fig, ax=ax_hist)

    fig.tight_layout()

    fig_name = format_fig_name(
        num=1,
        short=f"sensor_advance_retreat_bias_hist_AMSREminusSMMR_{YEARS.start}-{YEARS.stop - 1}",
    )
    fig_path = get_fig_path(
        project_root=PROJECT_ROOT, subfolder=SUBFOLDER, fig_name=fig_name,
    )
    save_and_upload(fig, fig_path, remote_root=REMOTE_ROOT, remote_subdir="")


if __name__ == "__main__":
    main()