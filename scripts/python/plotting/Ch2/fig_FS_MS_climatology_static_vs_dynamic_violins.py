#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Walsh-style violins: static vs dynamic FS/MS climatological DOY
by canonical sector.

Row 1: FS (Static & Dynamic)
Row 2: MS (Static & Dynamic)
"""

import sys
from pathlib import Path

import numpy as np
import xarray as xr
import pandas as pd
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
    load_dynamic_phase_climatology,
)

set_mpl_defaults()
sns.set_style("whitegrid")

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
SENSOR = "SMMR"
THR_PCT = 15
K_DAYS = 5

YEAR_MIN = 1979
YEAR_MAX = 2023

INPUT_ROOT_STATIC = PROJECT_ROOT_CLUSTER / "results" / f"{SENSOR}_phase"

SECTOR_FILE = PROJECT_ROOT_CLUSTER / "data" / "canonical_sectors.nc"

# Canonical sector labels
sector_labels = {
    1: "Amundsen–\nBellingshausen",
    2: "Weddell",
    3: "King Haakon VII",
    4: "East Antarctica",
    5: "Ross–\nAmundsen",
}
sector_ids = [1, 2, 3, 4, 5]

# ---------------------------------------------------------------------
# Helpers: static climatology from annual fields
# ---------------------------------------------------------------------
def compute_static_climatology(metric: str) -> xr.DataArray:
    """
    Compute climatological mean for static FS or MS:
      results/SMMR_phase/<metric>_thr15_k5/<metric>_YYYY.nc

    metric: "FS" or "MS"
    Returns DA[y,x] with mean over YEAR_MIN..YEAR_MAX.
    """
    subdir = f"{metric}_thr{THR_PCT}_k{K_DAYS}"
    folder = INPUT_ROOT_STATIC / subdir

    years = []
    das = []

    for y in range(YEAR_MIN, YEAR_MAX + 1):
        fpath = folder / f"{metric}_{y}.nc"
        if not fpath.exists():
            continue
        ds = xr.open_dataset(fpath)
        da = ds[metric].load()
        ds.close()
        das.append(da.expand_dims(year=[y]))
        years.append(y)

    if not das:
        raise FileNotFoundError(f"No static files found in {folder} for {metric}")

    da_all = xr.concat(das, dim="year")
    da_all = da_all.assign_coords(year=("year", years))

    clim = da_all.mean("year", skipna=True)
    return clim


# ---------------------------------------------------------------------
# Load sector mask
# ---------------------------------------------------------------------
ds_sect = xr.open_dataset(SECTOR_FILE)
sector_mask = ds_sect["sector_id"].values       # [y,x]
ocean_mask = ds_sect["valid_ocean"].astype(bool).values

# ---------------------------------------------------------------------
# Load climatologies: static + dynamic
# ---------------------------------------------------------------------
print("Computing static climatologies...")
fs_stat_clim = compute_static_climatology("FS")
ms_stat_clim = compute_static_climatology("MS")

print("Computing dynamic climatologies...")
fs_dyn_clim = load_dynamic_phase_climatology("FS", YEAR_MIN, YEAR_MAX)
ms_dyn_clim = load_dynamic_phase_climatology("MS", YEAR_MIN, YEAR_MAX)

# Sanity: same grid shape
assert fs_stat_clim.shape == fs_dyn_clim.shape == sector_mask.shape

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

        # Optional clipping to keep y-range tidy
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

# Match your existing map ranges
add_phase_records("FS", fs_stat_clim, fs_dyn_clim, 80, 240)
add_phase_records("MS", ms_stat_clim, ms_dyn_clim, 200, 360)

df = pd.DataFrame.from_records(records)

# ---------------------------------------------------------------------
# Plot violins
# ---------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, dpi=300)

for ax, phase_name, ylim in zip(
    axes,
    ["FS", "MS"],
    [(80, 240), (200, 360)],
):
    sub = df[df["phase"] == phase_name]

    sns.violinplot(
        data=sub,
        x="sector",
        y="doy",
        hue="method",
        split=True,         # Static + Dynamic in same violin per sector
        inner="quartile",
        cut=0,
        linewidth=0.8,
        ax=ax,
    )

    ax.set_ylabel("Day of year")
    ax.set_xlabel("")
    ax.set_ylim(*ylim)
    ax.set_title(f"{phase_name} climatology (static vs dynamic)")

    if phase_name == "FS":
        ax.legend(
            loc="upper left",
            frameon=True,
            facecolor="white",
            framealpha=0.8,
            title="Method",
        )
    else:
        ax.legend_.remove()

axes[-1].set_xlabel("Sector")

fig.tight_layout()

out_path = get_fig_path(
    PROJECT_ROOT_CLUSTER,
    subfolder="climatologies",
    fig_name="Fig_FS_MS_climatology_static_vs_dynamic_violins.png",
)

save_and_upload(
    fig,
    out_path,
    remote_root="gdrive:sea-ice-phase/Results/Ch2_Figures",
    remote_subdir="climatologies",
)
