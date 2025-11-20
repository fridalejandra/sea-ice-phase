#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fig_FS_MS_threshold_sensitivity_static_maps.py

FS/MS mean |Δ date| between low/high SIC thresholds and 15% threshold
for the static (slope) method, k=5 days, SMMR period.

Panels:
  (a) FS |Δ(thr_low − thr_ref)|
  (b) FS |Δ(thr_high − thr_ref)|
  (c) MS |Δ(thr_low − thr_ref)|
  (d) MS |Δ(thr_high − thr_ref)|
"""

import sys
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

HERE = Path(__file__).resolve().parent
PLOTTING_ROOT = HERE.parent

if str(PLOTTING_ROOT) not in sys.path:
    sys.path.insert(0, str(PLOTTING_ROOT))

from ch2_fig_utils import (  # noqa: E402
    set_mpl_defaults,
    format_fig_name,
    get_fig_path,
    save_and_upload,
    PROJECT_ROOT_CLUSTER,
)

set_mpl_defaults()

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
SENSOR      = "SMMR"
K_DAYS      = 5
THR_LOW     = 10
THR_REF     = 15
THR_HIGH    = 30
PERIOD      = 366.0
YEAR_MIN    = 1979
YEAR_MAX    = 2023

INPUT_ROOT = PROJECT_ROOT_CLUSTER / "results" / f"{SENSOR}_phase"

VMAX = 10.0  # 0–10 day colour scale

# ---------------------------------------------------------------------
# Loading + diffs (same as in ECDF script)
# ---------------------------------------------------------------------
def _year_from_fname(path: Path) -> int | None:
    name = path.name
    try:
        return int(name.split("_")[1].split(".")[0])
    except Exception:
        return None


def load_thr_dict(metric: str, thr_pct: int, kdays: int) -> dict[int, xr.DataArray]:
    subdir = f"{metric}_thr{thr_pct}_k{kdays}"
    folder = INPUT_ROOT / subdir
    files = sorted(folder.glob(f"{metric}_*.nc"))
    out = {}

    if not files:
        raise FileNotFoundError(f"No files in {folder} for metric={metric}, thr={thr_pct}")

    for f in files:
        yr = _year_from_fname(f)
        if yr is None:
            continue
        if yr < YEAR_MIN or yr > YEAR_MAX:
            continue
        ds = xr.open_dataset(f)
        da = ds[metric].load()
        ds.close()
        out[yr] = da

    if not out:
        raise FileNotFoundError(
            f"No usable years in {folder} for metric={metric}, thr={thr_pct}"
        )
    return out


def align_years(dicts: list[dict[int, xr.DataArray]]) -> list[int]:
    common = set.intersection(*[set(d.keys()) for d in dicts])
    years = sorted(common)
    if not years:
        raise ValueError("No overlapping years across thresholds.")
    return years


def stack_years(d: dict[int, xr.DataArray], years: list[int]) -> xr.DataArray:
    arrs = [d[y].expand_dims(year=[y]) for y in years]
    return xr.concat(arrs, dim="year")


def wrapped_diff(arr: np.ndarray, period: float = PERIOD) -> np.ndarray:
    return (arr + period / 2.0) % period - period / 2.0


def compute_thr_diff_means(metric: str) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Returns mean DOY-wrapped differences across years for:
      thr_low − thr_ref and thr_high − thr_ref.
    """
    d_low = load_thr_dict(metric, THR_LOW,  K_DAYS)
    d_ref = load_thr_dict(metric, THR_REF, K_DAYS)
    d_high = load_thr_dict(metric, THR_HIGH, K_DAYS)

    years = align_years([d_low, d_ref, d_high])

    A_low  = stack_years(d_low,  years)
    A_ref  = stack_years(d_ref,  years)
    A_high = stack_years(d_high, years)

    diff_low  = wrapped_diff((A_low  - A_ref).values)
    diff_high = wrapped_diff((A_high - A_ref).values)

    low_da = xr.DataArray(
        diff_low,
        coords=A_ref.coords,
        dims=A_ref.dims,
        name=f"{metric}_thr{THR_LOW}minus{THR_REF}",
    ).mean("year", skipna=True)

    high_da = xr.DataArray(
        diff_high,
        coords=A_ref.coords,
        dims=A_ref.dims,
        name=f"{metric}_thr{THR_HIGH}minus{THR_REF}",
    ).mean("year", skipna=True)

    return low_da, high_da

# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------
def make_polar_ax(fig, pos):
    proj = ccrs.SouthPolarStereo()
    ax = fig.add_subplot(pos, projection=proj)

    ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
    ax.add_feature(
        cfeature.LAND.with_scale("110m"),
        facecolor="0.85",
        edgecolor="0.6",
        linewidth=0.4,
        zorder=1,
    )
    ax.coastlines(linewidth=0.4, color="0.4", zorder=2)

    gl = ax.gridlines(draw_labels=False)
    gl.xlines = False
    gl.ylines = False

    ax.set_facecolor("white")
    return ax

# ---------------------------------------------------------------------
# MAIN plotting
# ---------------------------------------------------------------------
def plot_threshold_maps():
    print("\nComputing mean threshold differences for maps...")
    fs_low, fs_high = compute_thr_diff_means("FS")
    ms_low, ms_high = compute_thr_diff_means("MS")

    # Absolute magnitude, to match ECDFs of |Δ|
    fs_low_abs  = np.abs(fs_low)
    fs_high_abs = np.abs(fs_high)
    ms_low_abs  = np.abs(ms_low)
    ms_high_abs = np.abs(ms_high)

    # Get native x,y from one of the fields
    example_fs = next(iter(load_thr_dict("FS", THR_REF, K_DAYS).values()))
    x = example_fs["x"]
    y = example_fs["y"]

    proj = ccrs.SouthPolarStereo()
    fig = plt.figure(figsize=(8.0, 6.0))

    panels = [
        (f"FS {THR_LOW}–{THR_REF}%",  fs_low_abs,  221),
        (f"FS {THR_HIGH}–{THR_REF}%", fs_high_abs, 222),
        (f"MS {THR_LOW}–{THR_REF}%",  ms_low_abs,  223),
        (f"MS {THR_HIGH}–{THR_REF}%", ms_high_abs, 224),
    ]

    axes = []
    im_last = None
    for title, da_abs, code in panels:
        ax = make_polar_ax(fig, code)
        axes.append(ax)

        im_last = ax.pcolormesh(
            x,
            y,
            da_abs,
            transform=proj,
            cmap="viridis",
            vmin=0,
            vmax=VMAX,
            shading="auto",
        )
        ax.set_title(title, fontsize=9)

    # Shared colourbar
    cax = fig.add_axes([0.15, 0.08, 0.7, 0.03])
    cb = fig.colorbar(im_last, cax=cax, orientation="horizontal")
    cb.set_label("|Δ date| relative to 15% threshold (days)", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_visible(False)

    letters = ["(a)", "(b)", "(c)", "(d)"]
    for letter, ax in zip(letters, axes):
        ax.text(
            0.02,
            0.98,
            letter,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            fontweight="bold",
        )

    fig.tight_layout(rect=[0, 0.12, 1, 1])

    fig_name = format_fig_name(
        num=3,  # adjust numbering later
        short=f"threshold_FS_MS_static_{SENSOR}_thr{THR_LOW}_{THR_REF}_{THR_HIGH}",
    )

    out_path = get_fig_path(
        project_root=PROJECT_ROOT_CLUSTER,
        subfolder="sensitivity/threshold",
        fig_name=fig_name,
    )

    save_and_upload(
        fig,
        out_path,
        remote_root="gdrive:sea-ice-phase/Results/Ch2_Figures",
        remote_subdir="sensitivity/threshold",
    )


def main():
    set_mpl_defaults()
    plot_threshold_maps()


if __name__ == "__main__":
    main()
