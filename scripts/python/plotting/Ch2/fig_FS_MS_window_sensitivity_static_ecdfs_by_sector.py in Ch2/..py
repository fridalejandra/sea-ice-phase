#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------
# Make ch2_fig_utils importable
# -----------------------------------------------------------
HERE = Path(__file__).resolve().parent
PLOTTING_ROOT = HERE.parent

if str(PLOTTING_ROOT) not in sys.path:
    sys.path.insert(0, str(PLOTTING_ROOT))

from ch2_fig_utils import (
    set_mpl_defaults,
    get_fig_path,
    save_and_upload,
    PROJECT_ROOT_CLUSTER,
)

set_mpl_defaults()

# -----------------------------------------------------------
# Paths
# -----------------------------------------------------------
WINDOW_DIFF_DIR = (
    PROJECT_ROOT_CLUSTER
    / "results"
    / "sensitivity"
    / "SMMR_phase"
    / "SMMR_window_comparison"
)

# TODO: point this to your canonical sector mask
SECTOR_MASK_PATH = (
    PROJECT_ROOT_CLUSTER
    / "results"
    / "masks"
    / "sector_mask_canonical.nc"
)
SECTOR_VAR_NAME = "sector"  # adjust if different

sector_da = xr.open_dataarray(SECTOR_MASK_PATH)
sector_mask = sector_da[SECTOR_VAR_NAME].values  # 2D [y,x]

# Define which sector IDs and labels to use
sector_ids = [1, 2, 3, 4, 5]  # adjust to your mask
sector_labels = {
    1: "Weddell",
    2: "Indian",
    3: "Pacific",
    4: "Ross",
    5: "Bell–Amund.",
}

# -----------------------------------------------------------
# Load diff fields as |Δ|, keeping spatial structure
# -----------------------------------------------------------
def load_abs_diff_field(pattern: str) -> xr.DataArray:
    """
    Load |ΔDOY| as a DataArray [year, y, x] for a diff field.

    pattern: e.g. 'diff_advance_3minus5_*.nc'
    Assumes data variable has same base name, e.g. 'diff_advance_3minus5'.
    """
    fpaths = sorted(WINDOW_DIFF_DIR.glob(pattern))
    if not fpaths:
        raise FileNotFoundError(f"No files matching {pattern} in {WINDOW_DIFF_DIR}")

    ds = xr.open_mfdataset(fpaths, concat_dim="year", combine="nested")

    base = pattern.split("_*.nc")[0]  # "diff_advance_3minus5"
    if base not in ds.data_vars:
        raise KeyError(
            f"Variable '{base}' not found in dataset. "
            f"Available: {list(ds.data_vars)}"
        )

    da = ds[base]       # [year, y, x]
    return np.abs(da)   # keep as DataArray


adv_3v5_da = load_abs_diff_field("diff_advance_3minus5_*.nc")
adv_7v5_da = load_abs_diff_field("diff_advance_7minus5_*.nc")
ret_3v5_da = load_abs_diff_field("diff_retreat_3minus5_*.nc")
ret_7v5_da = load_abs_diff_field("diff_retreat_7minus5_*.nc")

# sanity: share y,x with sector mask
assert adv_3v5_da.sizes["y"] == sector_mask.shape[0]
assert adv_3v5_da.sizes["x"] == sector_mask.shape[1]

# -----------------------------------------------------------
# Sector-wise ECDF data
# -----------------------------------------------------------
def sector_values(da: xr.DataArray, sec_id: int) -> np.ndarray:
    """Flatten values for given sector over year,y,x."""
    mask = (sector_mask == sec_id)
    vals = da.values[:, mask]   # [year, n_pixels_in_sector]
    vals = vals.ravel()
    vals = vals[np.isfinite(vals)]
    return vals


def sector_combined_fsms(sec_id: int) -> tuple[np.ndarray, np.ndarray]:
    """
    For a sector ID, return combined FS+MS |Δ| values
    for 3v5 and 7v5 windows.
    """
    fs_3v5 = sector_values(adv_3v5_da, sec_id)
    fs_7v5 = sector_values(adv_7v5_da, sec_id)
    ms_3v5 = sector_values(ret_3v5_da, sec_id)
    ms_7v5 = sector_values(ret_7v5_da, sec_id)

    vals_3v5 = np.concatenate([fs_3v5, ms_3v5])
    vals_7v5 = np.concatenate([fs_7v5, ms_7v5])

    # Optional: clip extreme outliers (e.g. >30 days)
    vals_3v5 = vals_3v5[(vals_3v5 >= 0) & (vals_3v5 <= 30)]
    vals_7v5 = vals_7v5[(vals_7v5 >= 0) & (vals_7v5 <= 30)]

    return vals_3v5, vals_7v5


# -----------------------------------------------------------
# Plot sector ECDFs
# -----------------------------------------------------------
nsec = len(sector_ids)
ncols = 3
nrows = int(np.ceil(nsec / ncols))

fig, axes = plt.subplots(
    nrows,
    ncols,
    figsize=(3.2 * ncols, 3.0 * nrows),
    dpi=300,
    sharex=True,
    sharey=True,
)

axes = axes.ravel()
xmax = 30

for ax, sec_id in zip(axes, sector_ids):
    vals_3v5, vals_7v5 = sector_combined_fsms(sec_id)
    if vals_3v5.size == 0 or vals_7v5.size == 0:
        ax.set_visible(False)
        continue

    sns.ecdfplot(x=vals_3v5, ax=ax, label="3 vs 5 days")
    sns.ecdfplot(x=vals_7v5, ax=ax, label="7 vs 5 days")

    ax.set_xlim(0, xmax)
    ax.set_title(sector_labels.get(sec_id, f"Sector {sec_id}"))
    ax.grid(True, alpha=0.3)

# shared labels
for ax in axes[:nsec]:
    ax.set_xlabel("Absolute timing difference (days)")
    ax.set_ylabel("Cumulative fraction of pixels")

for ax in axes[nsec:]:
    ax.set_visible(False)

# single legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)

fig.suptitle("FS+MS window sensitivity by sector (static slope)", y=0.98)
fig.tight_layout(rect=[0, 0.06, 1, 0.96])

# -----------------------------------------------------------
# Save + upload
# -----------------------------------------------------------
out_path = get_fig_path(
    PROJECT_ROOT_CLUSTER,
    subfolder="sensitivity/window",
    fig_name="Fig_FS_MS_window_sensitivity_static_ecdfs_by_sector.png",
)

save_and_upload(
    fig,
    out_path,
    remote_root="gdrive:sea-ice-phase/Results/Ch2_Figures",
    remote_subdir="sensitivity/window",
)
