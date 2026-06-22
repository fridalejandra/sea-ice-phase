#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fig1_sensor_bias_AMSRE_minus_SMMR.py

Compare AMSRE vs SMMR phase dates (old static advance/retreat)
and produce a combined figure with:

  - climatological wrapped-bias maps (AMSRE − SMMR, days) for advance and retreat
  - a stacked histogram of the wrapped bias, showing both phases on the same axis

Uses ch2_fig_utils for style, naming, paths, and rclone upload.
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
# Make sure project root is on sys.path so "scripts.*" imports work
#   this assumes this file lives in:
#   sea-ice-phase/scripts/python/plotting/Ch2/
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[4]  # -> sea-ice-phase
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

# AMSRE period you care about (years that exist for BOTH AMSRE and SMMR)
YEARS = range(2012, 2024)  # 2012–2023 inclusive

# where the per-year phase files live (adjust AMSRE_DIR if needed)
SMMR_DIR = PROJECT_ROOT / "results" / "SMMR_phase"
AMSRE_DIR = PROJECT_ROOT / "results" / "AMSRE_phase"

# where to put figures (relative to PROJECT_ROOT and your rclone remote)
SUBFOLDER = ""
REMOTE_ROOT = "gdrive:sea-ice-phase/results/Ch2_Figures"


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------


def wrapped_difference(a, b, period=365.0):
    """Circular difference: (a - b) mapped into [-period/2, period/2].

    Example: 360 vs 5 becomes -10, not +355.
    """
    diff = (a - b + period / 2.0) % period - period / 2.0
    return diff


def plot_bias_map(ax, bias_da, vlim=20):
    """Plot a South Polar Stereo map of bias in days on the given axes.

    Parameters
    ----------
    ax : matplotlib Axes with a SouthPolarStereo projection
    bias_da : xr.DataArray
        Bias field [y, x] on the SMMR grid.
    vlim : float
        Symmetric colorbar limit in days.
    """
    data = bias_da.values
    x = bias_da["x"].values
    y = bias_da["y"].values

    proj = ccrs.SouthPolarStereo()

    im = ax.pcolormesh(
        x,
        y,
        data,
        transform=proj,  # data already in this projection
        cmap="RdBu_r",
        vmin=-vlim,
        vmax=+vlim,
        shading="auto",
    )

    ax.set_extent([-180, 180, -90, -50], crs=ccrs.PlateCarree())
    ax.set_facecolor("white")  # fully white background
    ax.coastlines(color="0.4", linewidth=0.5)

    # no title here – we’re keeping the figure clean
    return im



def compute_bias_for_phase(phase):
    """Compute annual and climatological wrapped bias for a given phase.

    Parameters
    ----------
    phase : {"advance", "retreat"}

    Returns
    -------
    bias_clim : xr.DataArray
        Climatological bias field [y, x].
    all_bias : np.ndarray
        Flattened 1D array of all bias values across years (NaNs removed).
    """
    yearly_bias_list = []
    all_bias_flat = []

    for year in YEARS:
        print(f"Processing {phase} for {year}")

        # old static variable names are like "advance_2012", "retreat_2012"
        varname = f"{phase}_{year}"

        smmr_path = SMMR_DIR / f"seaice_phases_SMMR_{year}.nc"
        amsre_path = AMSRE_DIR / f"seaice_phases_AMSRE_{year}.nc"

        ds_smmr = xr.open_dataset(smmr_path)
        ds_amsre = xr.open_dataset(amsre_path)

        smmr = ds_smmr[varname].load()   # [y_smmr, x_smmr]
        amsre = ds_amsre[varname].load()  # [y_amsre, x_amsre] on finer grid

        # coarsen AMSRE to SMMR resolution (assumes dims are "y", "x")
        ny, nx = amsre.shape
        if ny % 2 != 0:
            amsre = amsre.isel(y=slice(0, ny - 1))
        if nx % 2 != 0:
            amsre = amsre.isel(x=slice(0, nx - 1))

        amsre_coarse = amsre.coarsen(y=2, x=2, boundary="trim").mean()

        # align coordinates with SMMR grid
        amsre_coarse = amsre_coarse.assign_coords(x=smmr.x, y=smmr.y)

        # wrapped bias in days
        bias_vals = wrapped_difference(amsre_coarse.values, smmr.values, period=365.0)

        bias = xr.DataArray(
            data=bias_vals,
            coords={"y": smmr.y, "x": smmr.x},
            dims=("y", "x"),
            name="bias",
        )

        yearly_bias_list.append(bias)
        all_bias_flat.append(bias.values.ravel())

    # climatological bias (mean over years)
    bias_stack = xr.concat(yearly_bias_list, dim="year")
    bias_clim = bias_stack.mean("year", skipna=True)

    all_bias = np.concatenate(all_bias_flat)
    all_bias = all_bias[~np.isnan(all_bias)]

    return bias_clim, all_bias


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------


def main():
    set_mpl_defaults()
    sns.set_theme(style="ticks")

    # compute bias fields and flattened 1D arrays for both phases
    bias_clim_adv, all_bias_adv = compute_bias_for_phase("advance")
    bias_clim_ret, all_bias_ret = compute_bias_for_phase("retreat")

    # -----------------------------------------------------------------
    # Combined figure: two maps + stacked histogram
    # -----------------------------------------------------------------
    fig = plt.figure(figsize=(14, 5))
    proj = ccrs.SouthPolarStereo()
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.05, 1.2], wspace=0.3)

    ax_ret = fig.add_subplot(gs[0, 0], projection=proj)
    ax_adv = fig.add_subplot(gs[0, 1], projection=proj)
    ax_hist = fig.add_subplot(gs[0, 2])

    # Subplot labels
    ax_ret.text(0.02, 0.98, "(a)", transform=ax_ret.transAxes,
                ha="left", va="top", fontsize=12, fontweight="bold")

    ax_adv.text(0.02, 0.98, "(b)", transform=ax_adv.transAxes,
                ha="left", va="top", fontsize=12, fontweight="bold")

    ax_hist.text(0.02, 0.98, "(c)", transform=ax_hist.transAxes,
                 ha="left", va="top", fontsize=12, fontweight="bold")

    # maps
    title_years = f"{YEARS.start}–{YEARS.stop - 1}"

    im_ret = plot_bias_map(ax_ret, bias_clim_ret, vlim=20)
    im_adv = plot_bias_map(ax_adv, bias_clim_adv, vlim=20)

    # one shared horizontal colorbar for both maps
    cbar = fig.colorbar(
        im_ret,
        ax=[ax_ret, ax_adv],
        orientation="horizontal",
        pad=0.08,
        shrink=0.9,
    )
    cbar.set_label("Bias (AMSRE − SMMR, days)")

    # stacked histogram (advance + retreat on same axis)
    bias_all = np.concatenate([all_bias_adv, all_bias_ret])
    phase_all = (["Advance"] * len(all_bias_adv)) + (["Retreat"] * len(all_bias_ret))

    df = pd.DataFrame({"bias": bias_all, "phase": phase_all})

    sns.histplot(
        data=df,
        x="bias",
        hue="phase",
        multiple="stack",
        bins=np.arange(-40, 42, 2),
        stat="density",
        common_norm=False,  # <- per-phase densities
        edgecolor=".3",
        linewidth=0.5,
        ax=ax_hist,
    )

    ax_hist.set_ylim(0, 0.08)  # enough room for retreat’s ~0.06 peak

    # remove legend title ("phase" is obvious)
    leg = ax_hist.get_legend()
    if leg is not None:
        leg.set_title("")


    ax_hist.axvline(0, color="k", linewidth=0.8)
    ax_hist.set_xlabel("Bias (AMSRE − SMMR, days)")
    ax_hist.set_ylabel("Probability density")
    ax_hist.set_title("")  # no title
    ax_hist.set_xlim(-40, 40)
    ax_hist.set_xticks([-40, -20, 0, 20, 40])

    sns.despine(fig=fig, ax=ax_hist)

    fig.tight_layout()

    # save/upload
    fig_name = format_fig_name(
        num=8,  # adjust to your figure numbering if needed
        short=f"sensor_advance_retreat_bias_hist_AMSREminusSMMR_{YEARS.start}-{YEARS.stop - 1}",
    )

    fig_path = get_fig_path(
        project_root=PROJECT_ROOT,
        subfolder=SUBFOLDER,
        fig_name=fig_name,
    )

    save_and_upload(
        fig,
        fig_path,
        remote_root=REMOTE_ROOT,
        remote_subdir="",
    )


if __name__ == "__main__":
    main()
