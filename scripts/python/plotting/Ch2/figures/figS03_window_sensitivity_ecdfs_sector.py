#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
figS03_window_sensitivity_ecdfs_sector.py

Window sensitivity ECDFs by sector — FS and MS separately.
Static method, thr=15%, k=3,7 vs k=5. SMMR 1979-2024.

Layout: 2 rows (FS, MS) x 5 cols (one per sector)
"""

import sys
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns

HERE          = Path(__file__).resolve().parent
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
SENSOR     = "SMMR"
THRESH_PCT = 15
YEAR_MIN   = 1979
YEAR_MAX   = 2024
PERIOD     = 366.0
CLIP       = 30.0

INPUT_ROOT   = PROJECT_ROOT_CLUSTER / "data" / f"{SENSOR}_phase" / "static"
SECTOR_FILE  = PROJECT_ROOT_CLUSTER / "data" / "canonical_sectors.nc"

SECTOR_IDS = [1, 2, 3, 4, 5]
SECTOR_LABELS = {
    1: "AB",
    2: "Weddell",
    3: "KH VII",
    4: "E. Antarctica",
    5: "Ross",
}


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def _year_from_fname(path):
    try:
        return int(path.name.split("_")[1].split(".")[0])
    except Exception:
        return None


def load_window_dict(metric, kdays):
    subdir = f"thr{THRESH_PCT:02d}_k{kdays}"
    folder = INPUT_ROOT / subdir / metric
    files  = sorted(folder.glob(f"{metric}_*.nc"))
    out    = {}
    for f in files:
        yr = _year_from_fname(f)
        if yr is None or yr < YEAR_MIN or yr > YEAR_MAX:
            continue
        ds = xr.open_dataset(f)
        out[yr] = ds[metric].load()
        ds.close()
    return out


def align_years(*dicts):
    return sorted(set.intersection(*[set(d.keys()) for d in dicts]))


def stack_years(d, years):
    return xr.concat([d[y].expand_dims(year=[y]) for y in years], dim="year")


def wrapped_diff(arr, period=PERIOD):
    return (arr + period / 2.0) % period - period / 2.0


def compute_diff_stack(metric, k_test, sentinel):
    """Returns |diff| array (year, y, x) with sentinel masked."""
    d_test = load_window_dict(metric, k_test)
    d_ref  = load_window_dict(metric, 5)
    years  = align_years(d_test, d_ref)
    A_test = stack_years(d_test, years)
    A_ref  = stack_years(d_ref,  years)
    diff   = np.abs(wrapped_diff((A_test - A_ref).values))
    diff[:, sentinel] = np.nan
    return diff  # (year, y, x)


def sector_vals(diff_stack, sector_mask, sec_id):
    mask = (sector_mask == sec_id)
    vals = diff_stack[:, mask].ravel()
    vals = vals[np.isfinite(vals)]
    return vals[(vals >= 0) & (vals <= CLIP)]


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    ds_sec      = xr.open_dataset(SECTOR_FILE)
    sector_mask = ds_sec["sector_id"].values.astype(float)
    ocean_mask  = ds_sec["valid_ocean"].values.astype(bool)
    ds_sec.close()
    sector_mask[~ocean_mask] = np.nan

    sent_fs = get_sentinel_mask(PROJECT_ROOT_CLUSTER, "FS")
    sent_ms = get_sentinel_mask(PROJECT_ROOT_CLUSTER, "MS")
    sent_fs = sent_fs.values if hasattr(sent_fs, "values") else sent_fs
    sent_ms = sent_ms.values if hasattr(sent_ms, "values") else sent_ms

    print("Computing FS diffs...")
    fs_3v5 = compute_diff_stack("FS", 3, sent_fs)
    fs_7v5 = compute_diff_stack("FS", 7, sent_fs)
    print("Computing MS diffs...")
    ms_3v5 = compute_diff_stack("MS", 3, sent_ms)
    ms_7v5 = compute_diff_stack("MS", 7, sent_ms)

    nsec  = len(SECTOR_IDS)
    fig, axes = plt.subplots(2, nsec, figsize=(3.0 * nsec, 5.5),
                             dpi=300, sharex=True, sharey=True)

    for col, sec_id in enumerate(SECTOR_IDS):
        for row, (phase, d3, d7) in enumerate([
            ("FS", fs_3v5, fs_7v5),
            ("MS", ms_3v5, ms_7v5),
        ]):
            ax = axes[row, col]
            v3 = sector_vals(d3, sector_mask, sec_id)
            v7 = sector_vals(d7, sector_mask, sec_id)

            sns.ecdfplot(x=v3, ax=ax, label="3 vs 5")
            sns.ecdfplot(x=v7, ax=ax, label="7 vs 5")

            ax.set_xlim(0, CLIP)
            ax.grid(True, alpha=0.3)

            if row == 0:
                ax.set_title(SECTOR_LABELS[sec_id], fontsize=9, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{phase}\nCumul. fraction", fontsize=8)
            else:
                ax.set_ylabel("")
            if row == 1:
                ax.set_xlabel("|Δ date| (days)", fontsize=8)
            else:
                ax.set_xlabel("")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2,
               frameon=False, fontsize=9, title="Window comparison")
    fig.tight_layout(rect=[0, 0.06, 1, 1])

    out_path = get_fig_path(
        PROJECT_ROOT_CLUSTER,
        subfolder="",
        fig_name="FigS03_FS_MS_window_sensitivity_static_ecdfs_by_sector.png",
    )
    save_and_upload(
        fig, out_path,
        remote_root="gdrive:sea-ice-phase/results/Ch2_Figures",
        remote_subdir="",
    )


if __name__ == "__main__":
    main()