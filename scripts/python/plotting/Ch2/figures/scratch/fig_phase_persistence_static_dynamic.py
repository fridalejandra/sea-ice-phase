#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase persistence diagnostics for FS/MS, static vs dynamic.

Regimes:
  pre  = 1980–2015
  post = 2016–2023

Phases:
  FS = freeze-start (advance)
  MS = melt-start   (retreat)

Methods:
  - Static FS/MS from SMMR_phase files
  - Dynamic FS/MS from quantile_k5 (p0.7) files

Metrics:

  1) Lag-1 autocorrelation in time (year to year+1) for each phase:
       P_FS  = corr(FS_y, FS_{y+1})
       P_MS  = corr(MS_y, MS_{y+1})

     computed separately for static and dynamic, and for pre/post.

     We look at:
       ΔP_FS_static  = P_FS_post - P_FS_pre
       ΔP_FS_dynamic = ...
       ΔP_MS_static
       ΔP_MS_dynamic

  2) Cross-season phase persistence in each year:
       C_FS_MS = corr(FS_y, MS_y)  (across years)

     Again for static and dynamic, and for pre/post:
       ΔC_FS_MS_static  = C_post - C_pre
       ΔC_FS_MS_dynamic = ...

Figures:

  Fig 1: lag-1 autocorrelation changes
    - (a) ΔP_FS_static
    - (b) ΔP_FS_dynamic
    - (c) ΔP_MS_static
    - (d) ΔP_MS_dynamic

  Fig 2: FS–MS cross-season persistence changes
    - (a) ΔC_FS_MS_static
    - (b) ΔC_FS_MS_dynamic

Sector means could be added later if useful; this is a diagnostic
pass to inspect the spatial patterns first.
"""

import sys
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ---------------------------------------------------------------------
# ch2_fig_utils, paths, etc.
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

YEAR_MIN = 1980
YEAR_MAX = 2023

PRE_START_YEAR = 1980
PRE_END_YEAR = 2015
POST_START_YEAR = 2016
POST_END_YEAR = 2023

# Static FS/MS files
STATIC_DIR = PROJECT_ROOT / "results" / "SMMR_phase"

# Dynamic FS/MS dirs (quantile_k5, p0.7)
DYN_ROOT = (
    PROJECT_ROOT / "results" / "static_v2_slopeH" / "dynamic" / "quantile_k5"
)
DYN_DIR_FS = DYN_ROOT / "FS" / "p0.7"
DYN_DIR_MS = DYN_ROOT / "MS" / "p0.7"

# Sector / mask file
SECTOR_FILE = PROJECT_ROOT / "data" / "canonical_sectors.nc"

REMOTE_ROOT = "gdrive:sea-ice-phase/Results/Ch2_Figures"
SUBFOLDER = "phase_persistence"


# ---------------------------------------------------------------------
# Load sector mask
# ---------------------------------------------------------------------
ds_sect = xr.open_dataset(SECTOR_FILE)
sector_mask = ds_sect["sector_id"]
valid_ocean = ds_sect["valid_ocean"].astype(bool)
ds_sect.close()


# ---------------------------------------------------------------------
# Helpers to load annual FS/MS fields
# ---------------------------------------------------------------------
def _load_static_year(phase: str, year: int) -> xr.DataArray | None:
    """
    phase: 'FS' or 'MS'
    static file: seaice_phases_SMMR_YYYY.nc
      - FS -> advance_YYYY
      - MS -> retreat_YYYY
    """
    fpath = STATIC_DIR / f"seaice_phases_SMMR_{year}.nc"
    if not fpath.exists():
        return None

    ds = xr.open_dataset(fpath)

    if phase == "FS":
        var_prefix = "advance"
    elif phase == "MS":
        var_prefix = "retreat"
    else:
        ds.close()
        raise ValueError("phase must be 'FS' or 'MS'")

    varname = f"{var_prefix}_{year}"
    if varname not in ds:
        ds.close()
        return None

    da = ds[varname].load()
    ds.close()

    if not np.any(np.isfinite(da.values)):
        return None

    return da


def _load_dynamic_year(phase: str, year: int) -> xr.DataArray | None:
    """
    Dynamic FS/MS files:
      FS: .../quantile_k5/FS/p0.7/FS_YYYY.nc (var FS)
      MS: .../quantile_k5/MS/p0.7/MS_YYYY.nc (var MS)
    """
    if phase == "FS":
        ddir = DYN_DIR_FS
    elif phase == "MS":
        ddir = DYN_DIR_MS
    else:
        raise ValueError("phase must be 'FS' or 'MS'")

    fpath = ddir / f"{phase}_{year}.nc"
    if not fpath.exists():
        return None

    ds = xr.open_dataset(fpath)
    if phase not in ds:
        ds.close()
        return None

    da = ds[phase].load()
    ds.close()

    if not np.any(np.isfinite(da.values)):
        return None

    return da


def load_phase_stack(phase: str, method: str) -> xr.DataArray:
    """
    Load annual FS/MS stack for one phase and method.

    phase: 'FS' or 'MS'
    method: 'static' or 'dynamic'

    Returns:
        DataArray [year, y, x]
    """
    records = []
    years = []

    for y in range(YEAR_MIN, YEAR_MAX + 1):
        if method == "static":
            da = _load_static_year(phase, y)
        elif method == "dynamic":
            da = _load_dynamic_year(phase, y)
        else:
            raise ValueError("method must be 'static' or 'dynamic'")

        if da is None:
            continue

        records.append(da.expand_dims(year=[y]))
        years.append(y)

    if not records:
        raise ValueError(f"No data loaded for {phase} ({method})")

    out = xr.concat(records, dim="year")
    out = out.sortby("year")
    out.name = f"{phase}_{method}"
    return out


# ---------------------------------------------------------------------
# Lag-1 autocorrelation and FS–MS correlation
# ---------------------------------------------------------------------
def lag1_autocorr(da: xr.DataArray) -> xr.DataArray:
    """
    Lag-1 autocorrelation along 'year' dimension.

    Computes corr( X_y, X_{y+1} ) at each gridcell.
    """
    da1 = da.isel(year=slice(0, -1))
    da2 = da.isel(year=slice(1, None))

    # means
    m1 = da1.mean("year", skipna=True)
    m2 = da2.mean("year", skipna=True)

    a = da1 - m1
    b = da2 - m2

    cov = (a * b).mean("year", skipna=True)
    var1 = (a * a).mean("year", skipna=True)
    var2 = (b * b).mean("year", skipna=True)

    corr = cov / np.sqrt(var1 * var2)
    return corr


def fs_ms_corr(fs: xr.DataArray, ms: xr.DataArray) -> xr.DataArray:
    """
    Cross-season correlation between FS and MS along 'year'.

    corr( FS_y, MS_y ) at each gridcell.
    """
    # ensure same years
    common_years = np.intersect1d(fs["year"].values, ms["year"].values)
    fs = fs.sel(year=common_years)
    ms = ms.sel(year=common_years)

    fs_mean = fs.mean("year", skipna=True)
    ms_mean = ms.mean("year", skipna=True)

    fa = fs - fs_mean
    ma = ms - ms_mean

    cov = (fa * ma).mean("year", skipna=True)
    var_f = (fa * fa).mean("year", skipna=True)
    var_m = (ma * ma).mean("year", skipna=True)

    corr = cov / np.sqrt(var_f * var_m)
    return corr


def split_pre_post(da: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    pre = da.sel(year=slice(PRE_START_YEAR, PRE_END_YEAR))
    post = da.sel(year=slice(POST_START_YEAR, POST_END_YEAR))
    return pre, post


# ---------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------
def make_polar_ax(fig, gs, row, col, title=None):
    proj = ccrs.SouthPolarStereo()
    ax = fig.add_subplot(gs[row, col], projection=proj)
    ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="0.8", edgecolor="none", zorder=2)
    ax.set_facecolor("white")
    ax.gridlines(draw_labels=False, linewidth=0.0)
    if title:
        ax.set_title(title, fontsize=9, fontweight="bold")
    return ax


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    # -------------------------------------------------------------
    # Load FS/MS stacks
    # -------------------------------------------------------------
    print("Loading FS/MS static/dynamic stacks...")
    fs_static = load_phase_stack("FS", "static")
    ms_static = load_phase_stack("MS", "static")
    fs_dynamic = load_phase_stack("FS", "dynamic")
    ms_dynamic = load_phase_stack("MS", "dynamic")

    # mask
    fs_static = fs_static.where(valid_ocean)
    ms_static = ms_static.where(valid_ocean)
    fs_dynamic = fs_dynamic.where(valid_ocean)
    ms_dynamic = ms_dynamic.where(valid_ocean)

    # -------------------------------------------------------------
    # 1) Lag-1 autocorrelation (FS & MS)
    # -------------------------------------------------------------
    print("Computing lag-1 autocorrelation for FS/MS...")

    def compute_lag1_pre_post(da):
        pre, post = split_pre_post(da)
        P_pre = lag1_autocorr(pre)
        P_post = lag1_autocorr(post)
        dP = P_post - P_pre
        return P_pre, P_post, dP

    P_FS_stat_pre, P_FS_stat_post, dP_FS_stat = compute_lag1_pre_post(fs_static)
    P_MS_stat_pre, P_MS_stat_post, dP_MS_stat = compute_lag1_pre_post(ms_static)

    P_FS_dyn_pre, P_FS_dyn_post, dP_FS_dyn = compute_lag1_pre_post(fs_dynamic)
    P_MS_dyn_pre, P_MS_dyn_post, dP_MS_dyn = compute_lag1_pre_post(ms_dynamic)

    # -------------------------------------------------------------
    # 2) Cross-season FS–MS correlation, pre/post
    # -------------------------------------------------------------
    print("Computing FS–MS cross-season correlation...")

    def compute_fs_ms_pre_post(fs_da, ms_da):
        fs_pre, fs_post = split_pre_post(fs_da)
        ms_pre, ms_post = split_pre_post(ms_da)

        # ensure same years within each regime
        common_pre = np.intersect1d(fs_pre["year"].values, ms_pre["year"].values)
        common_post = np.intersect1d(fs_post["year"].values, ms_post["year"].values)

        fs_pre = fs_pre.sel(year=common_pre)
        ms_pre = ms_pre.sel(year=common_pre)
        fs_post = fs_post.sel(year=common_post)
        ms_post = ms_post.sel(year=common_post)

        C_pre = fs_ms_corr(fs_pre, ms_pre)
        C_post = fs_ms_corr(fs_post, ms_post)
        dC = C_post - C_pre
        return C_pre, C_post, dC

    C_stat_pre, C_stat_post, dC_stat = compute_fs_ms_pre_post(fs_static, ms_static)
    C_dyn_pre, C_dyn_post, dC_dyn = compute_fs_ms_pre_post(fs_dynamic, ms_dynamic)

    # -----------------------------------------------------------------
    # FIGURE 1: lag-1 autocorrelation changes ΔP
    # -----------------------------------------------------------------
    print("Plotting Fig 1: lag-1 phase persistence ΔP...")

    fig1 = plt.figure(figsize=(10, 8))
    gs1 = fig1.add_gridspec(2, 2, wspace=0.15, hspace=0.15)

    cmap_div = plt.get_cmap("RdBu_r")
    vmax_dp = 0.5  # correlation differences -0.5..0.5

    ax1 = make_polar_ax(
        fig1,
        gs1,
        0,
        0,
        "(a) ΔP_FS_static (post − pre)",
    )
    im1 = ax1.pcolormesh(
        dP_FS_stat["x"],
        dP_FS_stat["y"],
        dP_FS_stat,
        transform=ccrs.SouthPolarStereo(),
        cmap=cmap_div,
        vmin=-vmax_dp,
        vmax=vmax_dp,
        shading="auto",
    )

    ax2 = make_polar_ax(
        fig1,
        gs1,
        0,
        1,
        "(b) ΔP_FS_dynamic (post − pre)",
    )
    im2 = ax2.pcolormesh(
        dP_FS_dyn["x"],
        dP_FS_dyn["y"],
        dP_FS_dyn,
        transform=ccrs.SouthPolarStereo(),
        cmap=cmap_div,
        vmin=-vmax_dp,
        vmax=vmax_dp,
        shading="auto",
    )

    ax3 = make_polar_ax(
        fig1,
        gs1,
        1,
        0,
        "(c) ΔP_MS_static (post − pre)",
    )
    im3 = ax3.pcolormesh(
        dP_MS_stat["x"],
        dP_MS_stat["y"],
        dP_MS_stat,
        transform=ccrs.SouthPolarStereo(),
        cmap=cmap_div,
        vmin=-vmax_dp,
        vmax=vmax_dp,
        shading="auto",
    )

    ax4 = make_polar_ax(
        fig1,
        gs1,
        1,
        1,
        "(d) ΔP_MS_dynamic (post − pre)",
    )
    im4 = ax4.pcolormesh(
        dP_MS_dyn["x"],
        dP_MS_dyn["y"],
        dP_MS_dyn,
        transform=ccrs.SouthPolarStereo(),
        cmap=cmap_div,
        vmin=-vmax_dp,
        vmax=vmax_dp,
        shading="auto",
    )

    cax1 = fig1.add_axes([0.20, 0.05, 0.60, 0.025])
    cb1 = fig1.colorbar(im1, cax=cax1, orientation="horizontal")
    cb1.set_label("ΔP (post − pre) [lag-1 autocorrelation]", fontsize=9)
    cb1.ax.tick_params(labelsize=8)
    cb1.outline.set_visible(False)

    fig1.tight_layout(rect=[0.02, 0.08, 0.98, 0.98])

    out1 = get_fig_path(
        PROJECT_ROOT,
        subfolder=SUBFOLDER,
        fig_name="Fig_phase_persistence_lag1_FS_MS_static_dynamic_pre1980-2015_post2016-2023.png",
    )
    save_and_upload(
        fig1,
        out1,
        remote_root=REMOTE_ROOT,
        remote_subdir=SUBFOLDER,
    )

    # -----------------------------------------------------------------
    # FIGURE 2: FS–MS cross-season persistence changes ΔC
    # -----------------------------------------------------------------
    print("Plotting Fig 2: FS–MS cross-season persistence ΔC...")

    fig2 = plt.figure(figsize=(8, 4))
    gs2 = fig2.add_gridspec(1, 2, wspace=0.15)

    vmax_dc = 0.7

    ax5 = make_polar_ax(
        fig2,
        gs2,
        0,
        0,
        "(a) ΔC_FS_MS_static (post − pre)",
    )
    im5 = ax5.pcolormesh(
        dC_stat["x"],
        dC_stat["y"],
        dC_stat,
        transform=ccrs.SouthPolarStereo(),
        cmap=cmap_div,
        vmin=-vmax_dc,
        vmax=vmax_dc,
        shading="auto",
    )

    ax6 = make_polar_ax(
        fig2,
        gs2,
        0,
        1,
        "(b) ΔC_FS_MS_dynamic (post − pre)",
    )
    im6 = ax6.pcolormesh(
        dC_dyn["x"],
        dC_dyn["y"],
        dC_dyn,
        transform=ccrs.SouthPolarStereo(),
        cmap=cmap_div,
        vmin=-vmax_dc,
        vmax=vmax_dc,
        shading="auto",
    )

    cax2 = fig2.add_axes([0.20, 0.05, 0.60, 0.025])
    cb2 = fig2.colorbar(im5, cax=cax2, orientation="horizontal")
    cb2.set_label("Δ corr(FS, MS) (post − pre)", fontsize=9)
    cb2.ax.tick_params(labelsize=8)
    cb2.outline.set_visible(False)

    fig2.tight_layout(rect=[0.02, 0.08, 0.98, 0.98])

    out2 = get_fig_path(
        PROJECT_ROOT,
        subfolder=SUBFOLDER,
        fig_name="Fig_phase_FS_MS_crosscorr_static_dynamic_pre1980-2015_post2016-2023.png",
    )
    save_and_upload(
        fig2,
        out2,
        remote_root=REMOTE_ROOT,
        remote_subdir=SUBFOLDER,
    )

    print("Done.")


if __name__ == "__main__":
    main()
