#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
figS01_window_sensitivity_ecdfs.py

Circumpolar ECDFs of absolute timing differences between k=3,7 day
persistence windows and the reference k=5 day window, for static
FS and MS detection. SMMR 1979-2024.

Panels: single axis with 4 curves:
  FS 3 vs 5 days
  FS 7 vs 5 days
  MS 3 vs 5 days
  MS 7 vs 5 days
"""

import sys
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns

HERE         = Path(__file__).resolve().parent
PLOTTING_ROOT = HERE.parent
if str(PLOTTING_ROOT) not in sys.path:
    sys.path.insert(0, str(PLOTTING_ROOT))

from utils.plot_utils import (
    set_mpl_defaults,
    get_fig_path,
    save_and_upload,
    PROJECT_ROOT_CLUSTER,
    get_sentinel_mask,
)

set_mpl_defaults()

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
SENSOR    = "SMMR"
THRESH_PCT = 15
YEAR_MIN  = 1979
YEAR_MAX  = 2024
PERIOD    = 366.0
CLIP      = 30.0

INPUT_ROOT = PROJECT_ROOT_CLUSTER / "data" / f"{SENSOR}_phase" / "static"


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def _year_from_fname(path: Path) -> int | None:
    try:
        return int(path.name.split("_")[1].split(".")[0])
    except Exception:
        return None


def load_window_dict(metric: str, kdays: int) -> dict[int, xr.DataArray]:
    subdir = f"thr{THRESH_PCT:02d}_k{kdays}"
    folder = INPUT_ROOT / subdir / metric
    files  = sorted(folder.glob(f"{metric}_*.nc"))
    out    = {}
    if not files:
        raise FileNotFoundError(f"No files in {folder}")
    for f in files:
        yr = _year_from_fname(f)
        if yr is None or yr < YEAR_MIN or yr > YEAR_MAX:
            continue
        ds = xr.open_dataset(f)
        out[yr] = ds[metric].load()
        ds.close()
    return out


def align_years(*dicts):
    common = set.intersection(*[set(d.keys()) for d in dicts])
    return sorted(common)


def stack_years(d, years):
    return xr.concat([d[y].expand_dims(year=[y]) for y in years], dim="year")


def wrapped_diff(arr, period=PERIOD):
    return (arr + period / 2.0) % period - period / 2.0


def compute_ecdf_vals(metric: str, k_test: int, sentinel: np.ndarray) -> np.ndarray:
    """
    Compute |wrapped diff| between k_test and k=5, masked by sentinel.
    Returns flattened finite values clipped to [0, CLIP].
    """
    d_test = load_window_dict(metric, k_test)
    d_ref  = load_window_dict(metric, 5)
    years  = align_years(d_test, d_ref)
    A_test = stack_years(d_test, years)
    A_ref  = stack_years(d_ref,  years)
    diff   = np.abs(wrapped_diff((A_test - A_ref).values))  # (year, y, x)
    # mask sentinel pixels
    diff[:, sentinel] = np.nan
    vals = diff.ravel()
    vals = vals[np.isfinite(vals)]
    vals = vals[(vals >= 0) & (vals <= CLIP)]
    return vals


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    sent_fs = get_sentinel_mask(PROJECT_ROOT_CLUSTER, "FS")
    sent_ms = get_sentinel_mask(PROJECT_ROOT_CLUSTER, "MS")
    sent_fs = sent_fs.values if hasattr(sent_fs, "values") else sent_fs
    sent_ms = sent_ms.values if hasattr(sent_ms, "values") else sent_ms

    print("Computing FS 3v5...")
    fs_3v5 = compute_ecdf_vals("FS", 3, sent_fs)
    print("Computing FS 7v5...")
    fs_7v5 = compute_ecdf_vals("FS", 7, sent_fs)
    print("Computing MS 3v5...")
    ms_3v5 = compute_ecdf_vals("MS", 3, sent_ms)
    print("Computing MS 7v5...")
    ms_7v5 = compute_ecdf_vals("MS", 7, sent_ms)

    fig, ax = plt.subplots(figsize=(4.2, 3.2), dpi=300)

    sns.ecdfplot(x=fs_3v5, ax=ax, label="FS 3 vs 5 days")
    sns.ecdfplot(x=fs_7v5, ax=ax, label="FS 7 vs 5 days")
    sns.ecdfplot(x=ms_3v5, ax=ax, label="MS 3 vs 5 days")
    sns.ecdfplot(x=ms_7v5, ax=ax, label="MS 7 vs 5 days")

    ax.set_xlim(0, CLIP)
    ax.set_xlabel("|Δ date| (days)")
    ax.set_ylabel("Cumulative fraction of pixels")
    ax.grid(True, alpha=0.3)
    ax.legend(
        loc="center right",
        bbox_to_anchor=(0.95, 0.5),
        frameon=True,
        facecolor="white",
        framealpha=0.8,
        title="Window comparison",
    )

    fig.tight_layout()

    out_path = get_fig_path(
        PROJECT_ROOT_CLUSTER,
        subfolder="",
        fig_name="FigS01_FS_MS_window_sensitivity_static_ecdf_allcurves.png",
    )
    save_and_upload(
        fig, out_path,
        remote_root="gdrive:sea-ice-phase/results/Ch2_Figures",
        remote_subdir="",
    )


if __name__ == "__main__":
    main()