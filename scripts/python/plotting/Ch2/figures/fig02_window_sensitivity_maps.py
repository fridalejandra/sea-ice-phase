 #!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fig2_FS_MS_window_sensitivity_static_maps.py

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
    results/Ch2_Figures/sensitivity/window/Fig02_window_FS_MS_static_SMMR_thr15.png
    (also mirrored to gdrive:sea-ice-phase/results/Ch2_Figures/sensitivity/window/)
"""

import sys
from pathlib import Path

 import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ---------------------------------------------------------------------
# Ensure project root on sys.path so "scripts.*" imports work
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[5]  # -> sea-ice-phase
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.ch2_fig_utils import (  # noqa: E402
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

INPUT_ROOT = PROJECT_ROOT / "data" / f"{SENSOR}_phase" / "static"

REMOTE_ROOT = "gdrive:sea-ice-phase/results/Ch2_Figures"
SUBFOLDER   = ""

PERIOD = 366.0   # DOY wrap
VMAX   = 15.0    # colorbar half-range in days

YEAR_MIN = 1979
YEAR_MAX = 2024

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
    subdir = f"thr{THRESH_PCT}_k{kdays}"
    folder = INPUT_ROOT / subdir / metric
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

    ax.add_feature(
        cfeature.LAND.with_scale("110m"),
        facecolor="0.85",
        edgecolor="0.6",
        linewidth=0.4,
        zorder=1,
    )
    ax.coastlines(linewidth=0.4, color="0.4", zorder=2)

    # No gridlines for a clean look
    gl = ax.gridlines(draw_labels=False)
    gl.xlines = False
    gl.ylines = False

    ax.set_facecolor("white")

    return ax





def plot_window_diff_maps():
    """
    2x2 panel: FS/MS × (k=3−5, k=7−5) mean |Δ| timing differences.
    """
    print("\nComputing mean window differences for maps...")
    fs_d35, fs_d75 = compute_window_diff_means("FS")
    ms_d35, ms_d75 = compute_window_diff_means("MS")

    # Use absolute magnitude to match ECDFs of |Δ|
    fs_3v5_abs = np.abs(fs_d35)
    fs_7v5_abs = np.abs(fs_d75)
    ms_3v5_abs = np.abs(ms_d35)
    ms_7v5_abs = np.abs(ms_d75)
    # Mask sentinel pixels (open ocean / perennial ice assigned window-start DOY)
    from scripts.python.plotting.Ch2.utils.plot_utils import get_sentinel_mask
    fs_mask = get_sentinel_mask(PROJECT_ROOT, "FS")
    ms_mask = get_sentinel_mask(PROJECT_ROOT, "MS")
    fs_3v5_abs = fs_3v5_abs.where(~fs_mask)
    fs_7v5_abs = fs_7v5_abs.where(~fs_mask)
    ms_3v5_abs = ms_3v5_abs.where(~ms_mask)
    ms_7v5_abs = ms_7v5_abs.where(~ms_mask)

    # Native stereographic x,y grid
    example = next(iter(load_window_dict("FS", 5).values()))
    x = example["x"]
    y = example["y"]

    proj = ccrs.SouthPolarStereo()
    fig = plt.figure(figsize=(8.0, 6.0))

    panels = [
        ("FS 3–5", fs_3v5_abs, 221),
        ("FS 7–5", fs_7v5_abs, 222),
        ("MS 3–5", ms_3v5_abs, 223),
        ("MS 7–5", ms_7v5_abs, 224),
    ]

    vmax = 10.0  # 0–10 day scale

    im_last = None
    axes = []
    for title, da_abs, subplot_code in panels:
        ax = make_polar_ax(fig, subplot_code)
        axes.append(ax)

        im_last = ax.pcolormesh(
            x,
            y,
            da_abs,
            transform=proj,
            cmap="viridis",
            vmin=0,
            vmax=vmax,
            shading="auto",
        )
        ax.set_title(title, fontsize=9, fontweight="bold")

    # Shared colorbar
    cax = fig.add_axes([0.15, 0.08, 0.7, 0.03])
    cb = fig.colorbar(im_last, cax=cax, orientation="horizontal")
    cb.set_label("|Δ date| relative to 5-day window (days)", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_visible(False)

    # Optional panel letters
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
        num=2,  # adjust numbering in final manuscript
        short=f"window_FS_MS_static_{SENSOR}_thr{THRESH_PCT}_3v5_7v5_abs",
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
        remote_subdir="",
    )

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    set_mpl_defaults()
    plot_window_diff_maps()


if __name__ == "__main__":
    main()