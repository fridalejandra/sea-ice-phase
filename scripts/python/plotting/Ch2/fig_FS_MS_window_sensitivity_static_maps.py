#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fig_FS_MS_window_sensitivity_static_maps.py

Window sensitivity (static method, slope + persistence):

  - FS: mean (k=3 − 5) and (k=7 − 5) timing difference maps
  - MS: mean (k=3 − 5) and (k=7 − 5) timing difference maps

Differences are circular (DOY-wrapped) in days.

Input (existing static products):
    /user/geog/falejandraperez/sea-ice-phase/results/SMMR_phase/FS_thr15_k3/FS_YYYY.nc
    /user/geog/falejandraperez/sea-ice-phase/results/SMMR_phase/FS_thr15_k5/FS_YYYY.nc
    /user/geog/falejandraperez/sea-ice-phase/results/SMMR_phase/FS_thr15_k7/FS_YYYY.nc
    and similarly for MS.

Output:
    results/Ch2_Figures/sensitivity/window/Fig01_window_FS_MS_static_SMMR_thr15.png
    (also mirrored to gdrive:sea-ice-phase/Results/Ch2_Figures/sensitivity/window/)
"""

import sys
from pathlib import Path
from glob import glob

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ---------------------------------------------------------------------
# Ensure project root on sys.path so "scripts.*" imports work
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

SENSOR     = "SMMR"
THRESH_PCT = 15
WINDOWS    = [3, 5, 7]   # day windows

INPUT_ROOT = PROJECT_ROOT / "results" / f"{SENSOR}_phase"

REMOTE_ROOT = "gdrive:sea-ice-phase/Results/Ch2_Figures"
SUBFOLDER   = "sensitivity/window"

PERIOD = 366.0   # DOY wrap
VMAX   = 15.0    # colorbar half-range in days

YEAR_MIN = 1979
YEAR_MAX = 2023

# ---------------------------------------------------------------------
# HELPERS: loading + wrapped differences
# ---------------------------------------------------------------------

def _year_from_fname(path: Path) -> int | None:
    """Extract year from 'FS_1979.nc' etc."""
    name = path.name
    try:
        return int(name.split("_")[1].split(".")[0])
    except Exception:
        return None


def load_window_dict(metric: str, kdays: int) -> dict[int, xr.DataArray]:
    """
    Return {year: DataArray} for a given metric (FS/MS) and window (3/5/7).

    Looks under:
      results/SMMR_phase/<metric>_thr15_k<kdays>/<metric>_YYYY.nc
    """
    subdir = f"{metric}_thr{THRESH_PCT}_k{kdays}"
    folder = INPUT_ROOT / subdir
    files = sorted(Path(folder).glob(f"{metric}_*.nc"))
    out = {}

    if not files:
        raise FileNotFoundError(f"No files in {folder} for metric={metric}, k={kdays}")

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
        raise FileNotFoundError(f"No usable years in {folder} for metric={metric}, k={kdays}")

    return out


def align_years(dicts: list[dict[int, xr.DataArray]]) -> list[int]:
    common = set.intersection(*[set(d.keys()) for d in dicts])
    years = sorted(common)
    if not years:
        raise ValueError("No overlapping years across windows.")
    return years


def stack_years(d: dict[int, xr.DataArray], years: list[int]) -> xr.DataArray:
    """
    Stack dict of {year: DataArray} into DataArray with dim 'year'.
    """
    arrs = [d[y].expand_dims(year=[y]) for y in years]
    return xr.concat(arrs, dim="year")


def wrapped_diff(arr: np.ndarray, period: float = PERIOD) -> np.ndarray:
    """
    Circular difference mapping into [-period/2, period/2].
    """
    return (arr + period / 2.0) % period - period / 2.0


def compute_window_diff_means(metric: str) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Mean timing difference (k=3−5 and k=7−5) over all overlapping years.

    Returns
    -------
    d35_da, d75_da : DataArray
        Mean DOY-wrapped differences for (k=3−5) and (k=7−5), dims (y,x).
    """
    d3 = load_window_dict(metric, 3)
    d5 = load_window_dict(metric, 5)
    d7 = load_window_dict(metric, 7)

    years = align_years([d3, d5, d7])

    A3 = stack_years(d3, years)   # (year,y,x)
    A5 = stack_years(d5, years)
    A7 = stack_years(d7, years)

    d35 = wrapped_diff((A3 - A5).values)   # shape (year,y,x)
    d75 = wrapped_diff((A7 - A5).values)

    d35_da = xr.DataArray(
        d35,
        coords=A3.coords,
        dims=A3.dims,
        name=f"{metric}_d35",
    ).mean("year", skipna=True)

    d75_da = xr.DataArray(
        d75,
        coords=A3.coords,
        dims=A3.dims,
        name=f"{metric}_d75",
    ).mean("year", skipna=True)

    return d35_da, d75_da

# ---------------------------------------------------------------------
# HELPERS: plotting
# ---------------------------------------------------------------------

def make_polar_ax(fig, pos):
    proj = ccrs.SouthPolarStereo()
    ax = fig.add_subplot(pos, projection=proj)
    ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())

    # Clean, no black ocean
    ax.add_feature(cfeature.LAND, facecolor="0.8", edgecolor="0.6", zorder=1)
    ax.coastlines(linewidth=0.4, zorder=2)
    ax.gridlines(draw_labels=False, linewidth=0.3, color="0.5",
                 alpha=0.5, linestyle="--")
    return ax


def plot_window_diff_maps():
    """
    2x2 panel: FS/MS × (k=3−5, k=7−5) mean timing differences.
    """
    print("\nComputing mean window differences for maps...")
    fs_d35, fs_d75 = compute_window_diff_means("FS")
    ms_d35, ms_d75 = compute_window_diff_means("MS")

    # Get lon/lat if present; otherwise fall back to x/y
    example = next(iter(load_window_dict("FS", 5).values()))
    if {"lon", "lat"} <= set(example.coords):
        lons = example["lon"]
        lats = example["lat"]
    else:
        lons = example["x"]
        lats = example["y"]

    panels = [
        ("FS", fs_d35, "k = 3 − 5"),
        ("FS", fs_d75, "k = 7 − 5"),
        ("MS", ms_d35, "k = 3 − 5"),
        ("MS", ms_d75, "k = 7 − 5"),
    ]

    fig = plt.figure(figsize=(8.2, 6.0))
    axes = []

    for idx, (metric, da, label) in enumerate(panels, start=1):
        ax = make_polar_ax(fig, 220 + idx)  # 2x2 grid: 221, 222, 223, 224
        axes.append(ax)

        im = ax.pcolormesh(
            lons,
            lats,
            da,
            transform=ccrs.PlateCarree(),
            cmap="RdBu_r",
            vmin=-VMAX,
            vmax=+VMAX,
            shading="auto",
        )

        if metric == "FS":
            phase_name = "Freeze start (FS)"
        else:
            phase_name = "Melt start (MS)"

        ax.set_title(f"{phase_name}, {label}", fontsize=9)

    # Colorbar (shared)
    cax = fig.add_axes([0.15, 0.08, 0.7, 0.03])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("Timing difference relative to 5-day window (days)", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_visible(False)

    # Subfigure letters (a)–(d)
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

    fig.suptitle(
        f"Window sensitivity of FS/MS timing (static, thr={THRESH_PCT}%, {SENSOR})",
        fontsize=11,
    )

    fig.tight_layout(rect=[0, 0.14, 1, 0.94])

    # save + upload
    fig_name = format_fig_name(
        num=1,  # adjust once you finalize full figure ordering
        short=f"window_FS_MS_static_{SENSOR}_thr{THRESH_PCT}",
    )

    out_path = get_fig_path(
        project_root=PROJECT_ROOT,
        subfolder=SUBFOLDER,
        fig_name=fig_name,
    )

    save_and_upload(
        fig,
        out_path,
        remote_root=REMOTE_ROOT,
        remote_subdir=SUBFOLDER,
    )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    set_mpl_defaults()
    plot_window_diff_maps()


if __name__ == "__main__":
    main()
