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

from utils.plot_utils import (
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

SECTOR_MASK_PATH = (
    PROJECT_ROOT_CLUSTER
    / "data"
    / "canonical_sectors.nc"
)

sector_ds = xr.open_dataset(SECTOR_MASK_PATH)
print("Variables in sector file:", list(sector_ds.data_vars))

# Use sector_id as the mask
sector_mask = sector_ds["sector_id"].values          # 2D [y,x]

# Valid ocean mask (optional but sensible)
if "valid_ocean" in sector_ds:
    valid_ocean = sector_ds["valid_ocean"].values.astype(bool)
    sector_mask = np.where(valid_ocean, sector_mask, np.nan)
else:
    valid_ocean = ~np.isnan(sector_mask)

# Derive sector IDs from the mask (excluding NaNs/zeros if present)
sector_ids = sorted(
    int(s) for s in np.unique(sector_mask[valid_ocean])
    if np.isfinite(s) and s != 0
)

# Build labels from the name variables if you like
sector_labels = {}
for sid in sector_ids:
    name_var = f"sector_{sid}_name"
    if name_var in sector_ds:
        # these are probably 0-D string DataArrays
        sector_labels[sid] = str(sector_ds[name_var].values)
    else:
        sector_labels[sid] = f"Sector {sid}"

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
# -----------------------------------------------------------
# FS and MS separate ECDFs by sector
# -----------------------------------------------------------

nsec = len(sector_ids)
ncols = 3
nrows = 2 * int(np.ceil(nsec / ncols))  # FS rows first, MS rows second

fig, axes = plt.subplots(
    nrows,
    ncols,
    figsize=(3.2 * ncols, 2.8 * nrows),
    dpi=300,
    sharex=True,
    sharey=True,
)

axes = axes.ravel()
xmax = 30

# Row 1 → FS, Row 2 → MS
# Loop sectors twice: first FS then MS
plot_order = []
for phase in ["FS", "MS"]:
    for sec_id in sector_ids:
        plot_order.append((phase, sec_id))

for ax, (phase, sec_id) in zip(axes, plot_order):
    if phase == "FS":
        vals_3v5 = sector_values(adv_3v5_da, sec_id)
        vals_7v5 = sector_values(adv_7v5_da, sec_id)
    else:
        vals_3v5 = sector_values(ret_3v5_da, sec_id)
        vals_7v5 = sector_values(ret_7v5_da, sec_id)

    # Remove outliers > 30 days
    vals_3v5 = vals_3v5[(vals_3v5 >= 0) & (vals_3v5 <= 30)]
    vals_7v5 = vals_7v5[(vals_7v5 >= 0) & (vals_7v5 <= 30)]

    sns.ecdfplot(x=vals_3v5, ax=ax, label="3 vs 5 days")
    sns.ecdfplot(x=vals_7v5, ax=ax, label="7 vs 5 days")

    ax.set_xlim(0, xmax)
    title = f"{sector_labels[sec_id]} — {phase}"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

# Hide any extra blank panels
for ax in axes[len(plot_order):]:
    ax.set_visible(False)

# Shared x labels only on bottom row
for ax in axes[-ncols:]:
    ax.set_xlabel("Absolute timing difference (days)")

# Remove all y labels, add one shared label
for ax in axes:
    ax.set_ylabel("")
fig.text(0.04, 0.5, "Cumulative fraction of pixels",
         va="center", rotation="vertical", fontsize=10)

# Legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)

fig.tight_layout(rect=[0.06, 0.06, 1, 0.97])




out_path = get_fig_path(
    PROJECT_ROOT_CLUSTER,
    subfolder="",
    fig_name="FigS03_FS_MS_window_sensitivity_static_ecdfs_by_sector.png",
)

save_and_upload(
    fig,
    out_path,
    remote_root="gdrive:sea-ice-phase/results/Ch2_Figures",
    remote_subdir="",
)
