#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fig_FS_MS_threshold_sensitivity_static_ecdfs.py

Circumpolar ECDFs of absolute timing differences between low/high SIC
thresholds and the reference 15% threshold for static freeze start (FS)
and melt start (MS). Uses FS_thrXX_k5 / MS_thrXX_k5 daily products.

Thresholds are:
  THR_LOW  (e.g. 10%) vs THR_REF (15%)
  THR_HIGH (e.g. 20%) vs THR_REF (15%)
"""

import sys
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------
# Make ch2_fig_utils importable
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

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
SENSOR      = "SMMR"
K_DAYS      = 5          # window for these threshold tests
THR_LOW     = 10
THR_REF     = 15
THR_HIGH    = 30
PERIOD      = 366.0      # DOY wrap
YEAR_MIN    = 1979
YEAR_MAX    = 2023

INPUT_ROOT = PROJECT_ROOT_CLUSTER / "results" / f"{SENSOR}_phase"

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


def load_thr_dict(metric: str, thr_pct: int, kdays: int) -> dict[int, xr.DataArray]:
    """
    Return {year: DataArray} for a given metric (FS/MS) and threshold (10/15/20).

    Looks under:
      results/SMMR_phase/<metric>_thr<thr_pct>_k<kdays>/<metric>_YYYY.nc
    """
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
    """Circular difference mapping into [-period/2, period/2]."""
    return (arr + period / 2.0) % period - period / 2.0


def compute_thr_diff_arrays(metric: str) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Compute DOY-wrapped differences for low-ref and high-ref thresholds.

    Returns
    -------
    d_low_ref, d_high_ref : DataArray [year,y,x]
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

    d_low_da = xr.DataArray(
        diff_low,
        coords=A_ref.coords,
        dims=A_ref.dims,
        name=f"{metric}_thr{THR_LOW}minus{THR_REF}",
    )
    d_high_da = xr.DataArray(
        diff_high,
        coords=A_ref.coords,
        dims=A_ref.dims,
        name=f"{metric}_thr{THR_HIGH}minus{THR_REF}",
    )

    return d_low_da, d_high_da


def clip_0_30(arr: np.ndarray, clip: float = 30.0) -> np.ndarray:
    arr = arr[np.isfinite(arr)]
    return arr[(arr >= 0) & (arr <= clip)]

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    # FS
    fs_low_da, fs_high_da = compute_thr_diff_arrays("FS")
    ms_low_da, ms_high_da = compute_thr_diff_arrays("MS")

    fs_low_vals  = clip_0_30(np.abs(fs_low_da.values).ravel())
    fs_high_vals = clip_0_30(np.abs(fs_high_da.values).ravel())
    ms_low_vals  = clip_0_30(np.abs(ms_low_da.values).ravel())
    ms_high_vals = clip_0_30(np.abs(ms_high_da.values).ravel())

    clip = 30.0
    fig, ax = plt.subplots(figsize=(4.2, 3.2), dpi=300)

    # FS curves
    sns.ecdfplot(x=fs_low_vals,  ax=ax, label=f"FS {THR_LOW} vs {THR_REF}%")
    sns.ecdfplot(x=fs_high_vals, ax=ax, label=f"FS {THR_HIGH} vs {THR_REF}%")

    # MS curves
    sns.ecdfplot(x=ms_low_vals,  ax=ax, label=f"MS {THR_LOW} vs {THR_REF}%")
    sns.ecdfplot(x=ms_high_vals, ax=ax, label=f"MS {THR_HIGH} vs {THR_REF}%")

    ax.set_xlim(0, clip)
    ax.set_xlabel("|Δ date| (days)")
    ax.set_ylabel("Cumulative fraction of pixels")
    ax.grid(True, alpha=0.3)

    ax.legend(
        loc="center right",
        bbox_to_anchor=(0.95, 0.5),
        frameon=True,
        facecolor="white",
        framealpha=0.8,
        title="Threshold comparison",
    )

    fig.tight_layout()

    out_path = get_fig_path(
        PROJECT_ROOT_CLUSTER,
        subfolder="sensitivity/threshold",
        fig_name="Fig_FS_MS_threshold_sensitivity_static_ecdf_allcurves.png",
    )

    save_and_upload(
        fig,
        out_path,
        remote_root="gdrive:sea-ice-phase/Results/Ch2_Figures",
        remote_subdir="sensitivity/threshold",
    )


if __name__ == "__main__":
    main()
