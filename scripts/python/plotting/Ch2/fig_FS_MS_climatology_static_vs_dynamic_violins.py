#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Walsh-style violins: static vs dynamic FS/MS climatological DOY by sector.

Static:
  results/SMMR_phase/seaice_phases_SMMR_YYYY.nc
    - advance  -> FS_static
    - retreat  -> MS_static

Dynamic:
  results/static_v2_slopeH/dynamic/quantile_k5/FS/p0.7/FS_YYYY.nc
  results/static_v2_slopeH/dynamic/quantile_k5/MS/p0.7/MS_YYYY.nc
    - FS or MS variable inside
"""

import sys
from pathlib import Path

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------
# ch2_fig_utils
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
sns.set_style("whitegrid")

_base = sns.color_palette("colorblind")
METHOD_PALETTE = [_base[0], _base[2]]   # blue (Static) + green (Dynamic)

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
YEAR_MIN = 1980
YEAR_MAX = 2023

STATIC_DIR = PROJECT_ROOT_CLUSTER / "results" / "SMMR_phase"
DYN_ROOT = (
    PROJECT_ROOT_CLUSTER
    / "results"
    / "static_v2_slopeH"
    / "dynamic"
    / "quantile_k5"
)

# dynamic dirs:
DYN_DIR_FS = DYN_ROOT / "FS" / "p0.7"
DYN_DIR_MS = DYN_ROOT / "MS" / "p0.7"

SECTOR_FILE = PROJECT_ROOT_CLUSTER / "data" / "canonical_sectors.nc"

sector_labels = {
    1: "Amundsen–\nBellingshausen",
    2: "Weddell",
    3: "King Haakon VII",
    4: "East Antarctica",
    5: "Ross–\nAmundsen",
}
sector_ids = [1, 2, 3, 4, 5]

# ---------------------------------------------------------------------
# Sector mask
# ---------------------------------------------------------------------
ds_sect = xr.open_dataset(SECTOR_FILE)
sector_mask = ds_sect["sector_id"].values
ocean_mask = ds_sect["valid_ocean"].astype(bool).values

# Sanity: we’ll check shapes later once we’ve loaded a phase field
# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _load_static_year(phase: str, year: int) -> xr.DataArray | None:
    """
    phase: 'FS' or 'MS'
    static file: seaice_phases_SMMR_YYYY.nc
    variables: advance_YYYY, retreat_YYYY
    """
    fpath = STATIC_DIR / f"seaice_phases_SMMR_{year}.nc"
    if not fpath.exists():
        return None

    ds = xr.open_dataset(fpath)

    if phase == "FS":
        var_prefix = "advance"
    elif phase == "MS":
        var_prefix = "retreat"
    else:
        ds.close()
        raise ValueError("phase must be 'FS' or 'MS'")

    varname = f"{var_prefix}_{year}"

    # If variable missing → skip quietly
    if varname not in ds:
        ds.close()
        return None

    da = ds[varname].load()
    ds.close()

    # If all NaN → skip
    if not np.any(np.isfinite(da.values)):
        return None

    return da



def _load_dynamic_year(phase: str, year: int) -> xr.DataArray:
    """
    Dynamic FS/MS files in:
      FS: .../quantile_k5/FS/p0.7/FS_YYYY.nc
      MS: .../quantile_k5/MS/p0.7/MS_YYYY.nc
    """
    if phase == "FS":
        ddir = DYN_DIR_FS
    elif phase == "MS":
        ddir = DYN_DIR_MS
    else:
        raise ValueError("phase must be 'FS' or 'MS'")

    fpath = ddir / f"{phase}_{year}.nc"
    if not fpath.exists():
        return None

    ds = xr.open_dataset(fpath)
    if phase not in ds:
        raise KeyError(f"{phase} not in {fpath}; vars={list(ds.data_vars)}")

    da = ds[phase].load()
    ds.close()
    return da


def compute_climatologies_for_phase(phase: str) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Compute static and dynamic climatologies for one phase (FS or MS),
    using ONLY years where *both* exist.
    Returns:
        static_clim, dynamic_clim  (each [y,x])
    """
    static_dict = {}
    dyn_dict = {}

    for y in range(YEAR_MIN, YEAR_MAX + 1):
        da_s = _load_static_year(phase, y)
        da_d = _load_dynamic_year(phase, y)
        if da_s is None or da_d is None:
            continue

        static_dict[y] = da_s
        dyn_dict[y] = da_d

    common_years = sorted(set(static_dict.keys()) & set(dyn_dict.keys()))
    if not common_years:
        raise ValueError(f"No overlapping years for phase {phase}")

    # stack and mean
    stat_list = [static_dict[y].expand_dims(year=[y]) for y in common_years]
    dyn_list = [dyn_dict[y].expand_dims(year=[y]) for y in common_years]

    stat_all = xr.concat(stat_list, dim="year")
    dyn_all = xr.concat(dyn_list, dim="year")

    stat_clim = stat_all.mean("year", skipna=True)
    dyn_clim = dyn_all.mean("year", skipna=True)

    return stat_clim, dyn_clim


# ---------------------------------------------------------------------
# Compute climatologies
# ---------------------------------------------------------------------
print("Computing FS static/dynamic climatologies...")
fs_stat_clim, fs_dyn_clim = compute_climatologies_for_phase("FS")

print("Computing MS static/dynamic climatologies...")
ms_stat_clim, ms_dyn_clim = compute_climatologies_for_phase("MS")

# Check grid matches sector mask
if fs_stat_clim.shape != sector_mask.shape:
    raise ValueError(
        f"FS climatology shape {fs_stat_clim.shape} "
        f"does not match sector mask {sector_mask.shape}"
    )

# ---------------------------------------------------------------------
# Build DataFrame for violins
# ---------------------------------------------------------------------
records = []


def add_phase_records(phase_name, da_static, da_dynamic, ylim_min, ylim_max):
    for sec in sector_ids:
        mask = (sector_mask == sec) & ocean_mask

        stat_vals = da_static.where(mask).values.ravel()
        dyn_vals = da_dynamic.where(mask).values.ravel()

        stat_vals = stat_vals[np.isfinite(stat_vals)]
        dyn_vals = dyn_vals[np.isfinite(dyn_vals)]

        # Optional: clip to tidy range
        stat_vals = stat_vals[(stat_vals >= ylim_min) & (stat_vals <= ylim_max)]
        dyn_vals = dyn_vals[(dyn_vals >= ylim_min) & (dyn_vals <= ylim_max)]

        for v in stat_vals:
            records.append(
                {
                    "phase": phase_name,
                    "sector": sector_labels[sec],
                    "method": "Static",
                    "doy": float(v),
                }
            )
        for v in dyn_vals:
            records.append(
                {
                    "phase": phase_name,
                    "sector": sector_labels[sec],
                    "method": "Dynamic",
                    "doy": float(v),
                }
            )


# Your existing ranges
add_phase_records("FS", fs_stat_clim, fs_dyn_clim, 80, 240)
add_phase_records("MS", ms_stat_clim, ms_dyn_clim, 200, 360)

df = pd.DataFrame.from_records(records)

# ---------------------------------------------------------------------
# Plot violins (updated)
# ---------------------------------------------------------------------

sns.set(style="whitegrid")

# Light, complementary colors
palette = {
    "Static":  "#8ecae6",
    "Dynamic": "#ffb703",
}

sector_order = [
    "Amundsen–Bellingshausen",
    "Weddell",
    "King Haakon VII",
    "East Antarctica",
    "Ross–Amundsen",
]

fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, dpi=300)

# Panel letters + phase names
titles = {
    "FS": "(a) FS climatology",
    "MS": "(b) MS climatology",
}



for ax, phase_name in zip(axes, ["FS", "MS"]):

    sub = df[df["phase"] == phase_name]

    sns.violinplot(
        data=sub,
        x="sector",
        y="doy",
        hue="method",
        order=sector_order,
        palette=palette,
        split=True,
        inner="quartile",      # <-- shows median + IQR
        linewidth=1,
        cut=0,
        ax=ax,
        fill=False
    )

    ax.set_title(titles[phase_name], fontweight="bold", pad=8)
    ax.set_ylabel("Day of year")
    ax.tick_params(axis="x", rotation=0)

# Only show legend once
handles, labels = axes[0].get_legend_handles_labels()
axes[0].legend(handles, labels, title="", loc="upper right", frameon=True)
axes[1].get_legend().remove()

axes[1].set_xlabel("Sector")

fig.tight_layout()

out_path = get_fig_path(
    PROJECT_ROOT_CLUSTER,
    subfolder="climatology ",
    fig_name="Fig_FS_MS_climatology_static_vs_dynamic_violins.png",
)


save_and_upload(
    fig,
    out_path,
    remote_root="gdrive:sea-ice-phase/Results/Ch2_Figures",
    remote_subdir="climatology",
)
