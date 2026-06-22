#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path

import numpy as np
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# Make ch2_fig_utils importable (lives one level up: .../plotting)
# -----------------------------------------------------------
HERE = Path(__file__).resolve().parent
PLOTTING_ROOT = HERE.parent  # .../scripts/python/plotting

if str(PLOTTING_ROOT) not in sys.path:
    sys.path.insert(0, str(PLOTTING_ROOT))

from utils.plot_utils import (
    set_mpl_defaults,
    get_fig_path,
    save_and_upload,
    PROJECT_ROOT_CLUSTER,
)

set_mpl_defaults()

# -----------------------------------------------------------
# Paths and loading of diff files
# -----------------------------------------------------------
WINDOW_DIFF_DIR = (
    PROJECT_ROOT_CLUSTER
    / "results"
    / "sensitivity"
    / "SMMR_phase"
    / "SMMR_window_comparison"
)


def load_abs_diffs(pattern: str) -> np.ndarray:
    """
    Load |ΔDOY| values across all years for a given diff field.

    pattern: e.g. 'diff_advance_3minus5_*.nc'

    Assumes the data variable inside the file has the same base name
    as the pattern prefix, e.g. 'diff_advance_3minus5'.
    """
    fpaths = sorted(WINDOW_DIFF_DIR.glob(pattern))
    if not fpaths:
        raise FileNotFoundError(f"No files matching {pattern} in {WINDOW_DIFF_DIR}")

    ds = xr.open_mfdataset(fpaths, concat_dim="year", combine="nested")

    # Derive varname from pattern, e.g. "diff_advance_3minus5_*.nc" -> "diff_advance_3minus5"
    base = pattern.split("_*.nc")[0]
    if base not in ds.data_vars:
        raise KeyError(f"Variable '{base}' not found in dataset. Available: {list(ds.data_vars)}")

    da = ds[base]
    vals = np.abs(da.values).ravel()
    vals = vals[np.isfinite(vals)]
    return vals


adv_3v5_vals = load_abs_diffs("diff_advance_3minus5_*.nc")
adv_7v5_vals = load_abs_diffs("diff_advance_7minus5_*.nc")
ret_3v5_vals = load_abs_diffs("diff_retreat_3minus5_*.nc")
ret_7v5_vals = load_abs_diffs("diff_retreat_7minus5_*.nc")

print(WINDOW_DIFF_DIR)
print(list(WINDOW_DIFF_DIR.glob("diff_advance_3minus5_*.nc"))[:3])

# -----------------------------------------------------------
# Build combined ECDF: FS & MS, 3v5 and 7v5
# -----------------------------------------------------------

# Optionally clip insane tails (e.g. due to missing data artefacts)
clip = 30.0

def clip_0_30(arr):
    arr = arr[np.isfinite(arr)]
    return arr[(arr >= 0) & (arr <= clip)]

adv_3v5_vals = clip_0_30(adv_3v5_vals)
adv_7v5_vals = clip_0_30(adv_7v5_vals)
ret_3v5_vals = clip_0_30(ret_3v5_vals)
ret_7v5_vals = clip_0_30(ret_7v5_vals)

fig, ax = plt.subplots(figsize=(4.2, 3.2), dpi=300)

# FS curves
sns.ecdfplot(x=adv_3v5_vals, ax=ax, label="FS 3 vs 5 days")
sns.ecdfplot(x=adv_7v5_vals, ax=ax, label="FS 7 vs 5 days")

# MS curves
sns.ecdfplot(x=ret_3v5_vals, ax=ax, label="MS 3 vs 5 days")
sns.ecdfplot(x=ret_7v5_vals, ax=ax, label="MS 7 vs 5 days")

ax.set_xlim(0, clip)
ax.set_xlabel("|Δ date| (days)")
ax.set_ylabel("Cumulative fraction of pixels")
ax.grid(True, alpha=0.3)

# Put legend outside if crowded
ax.legend(
    loc="center right",
    bbox_to_anchor=(0.95, 0.5),
    frameon=True,
    facecolor="white",
    framealpha=0.8,
)


fig.tight_layout()

out_path = get_fig_path(
    PROJECT_ROOT_CLUSTER,
    subfolder="",
    fig_name="Fig_FS_MS_window_sensitivity_static_ecdf_allcurves.png",
)

save_and_upload(
    fig,
    out_path,
    remote_root="gdrive:sea-ice-phase/results/Ch2_Figures",
    remote_subdir="",
)
