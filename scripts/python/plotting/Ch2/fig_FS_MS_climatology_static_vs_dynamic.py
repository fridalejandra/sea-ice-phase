#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fig_FS_MS_climatology_static_vs_dynamic.py

Compare static vs dynamic FS/MS climatologies:

  - FS: static, dynamic, dynamic − static
  - MS: static, dynamic, dynamic − static

Static files:
    /user/geog/falejandraperez/sea-ice-phase/results/SMMR_phase/FS_thr15_k5/FS_YYYY.nc
    /user/geog/falejandraperez/sea-ice-phase/results/SMMR_phase/MS_thr15_k5/MS_YYYY.nc

Dynamic files (quantile p=0.7, k=5):
    /user/geog/falejandraperez/sea-ice-phase/results/static_v2_slopeH/dynamic/quantile_k5/FS/p0.7/FS_YYYY.nc
    /user/geog/falejandraperez/sea-ice-phase/results/static_v2_slopeH/dynamic/quantile_k5/MS/p0.7/MS_YYYY.nc
"""

import sys
from pathlib import Path
from glob import glob

import numpy as np
import xarray as xr

# ensure project root on sys.path so "scripts.*" imports work
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.python.plotting.ch2_fig_utils import (  # noqa: E402
    set_mpl_defaults,
    format_fig_name,
    get_fig_path,
    save_and_upload,
    plot_phase_comparison_map,
)


# ---------------------------------------------------------------------
# PATH CONFIG (EDIT ONLY IF YOU MOVE DATA)
# ---------------------------------------------------------------------

STATIC_ROOT = "/user/geog/falejandraperez/sea-ice-phase/results/SMMR_phase"

# Dynamic FS/MS (quantile p=0.7, k=5, from your dynamic script)
DYN_ROOT   = "/user/geog/falejandraperez/sea-ice-phase/results/static_v2_slopeH/dynamic"
DYN_SCHEME = "quantile_k5"
DYN_TAG    = "p0.7"

REMOTE_ROOT = "gdrive:sea-ice-phase/Results/Ch2_Figures"
SUBFOLDER   = "climatology"

PHASES = ["FS", "MS"]
YEAR_START = 1979
YEAR_END   = 2023


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def load_phase_climatology(phase: str, mode: str, year_start: int, year_end: int) -> xr.DataArray:
    """
    Load FS/MS files for given mode and return climatological mean over years.

    mode:
      - "static": results/SMMR_phase/<phase>_thr15_k5/<phase>_YYYY.nc
      - "dynamic": results/static_v2_slopeH/dynamic/quantile_k5/<phase>/p0.7/<phase>_YYYY.nc

    Assumes variable is named <phase> in each file.
    """
    if mode == "static":
        phase_dir = Path(STATIC_ROOT) / f"{phase}_thr15_k5"
        pattern = str(phase_dir / f"{phase}_*.nc")
    elif mode == "dynamic":
        phase_dir = Path(DYN_ROOT) / DYN_SCHEME / phase / DYN_TAG
        pattern = str(phase_dir / f"{phase}_*.nc")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    files = sorted(glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found for phase={phase}, mode={mode}, pattern={pattern}")

    years = []
    for f in files:
        base = Path(f).name  # e.g. "FS_1979.nc"
        try:
            y = int(base.split("_")[1].split(".")[0])
            years.append(y)
        except Exception:
            continue

    years = np.array(years)
    mask = (years >= year_start) & (years <= year_end)
    if not mask.any():
        raise ValueError(f"No years in [{year_start}, {year_end}] for phase={phase}, mode={mode}")

    files_sel = [f for f, m in zip(files, mask) if m]
    years_sel = years[mask]

    ds = xr.open_mfdataset(files_sel, combine="nested", concat_dim="year")
    ds = ds.assign_coords(year=("year", years_sel))

    da = ds[phase]
    clim = da.mean("year", skipna=True)

    # Keep lon/lat if present, otherwise just pass x/y
    return clim


def freeze_label(phase: str) -> str:
    """
    Map phase to human label for colorbar.
    FS -> Freeze start
    MS -> Melt start
    """
    if phase == "FS":
        return "Freeze start"
    elif phase == "MS":
        return "Melt start"
    else:
        return phase


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    set_mpl_defaults()

    # We assume lon/lat/x/y are identical between static and dynamic grids.
    # Grab coords from one example file.
    example_file = Path(STATIC_ROOT) / "FS_thr15_k5" / "FS_1979.nc"
    example = xr.open_dataset(example_file)
    # This will work whether you have lon/lat or just x/y; plot_phase_comparison_map
    # only needs lons/lats, so if you *don't* have them, we might need to adapt.
    # If lon/lat aren't there, fallback to x/y.
    if {"lon", "lat"} <= set(example.coords):
        lons = example["lon"]
        lats = example["lat"]
    else:
        # If no lon/lat, we still call plot_phase_comparison_map with x/y
        # by pretending x,y are "lon,lat". If that breaks, we can revert to native x/y plotting.
        lons = example["x"]
        lats = example["y"]

    for phase in PHASES:
        print(f"Processing climatology for {phase}")

        clim_static  = load_phase_climatology(phase, "static",  YEAR_START, YEAR_END)
        clim_dynamic = load_phase_climatology(phase, "dynamic", YEAR_START, YEAR_END)

        label = f"{freeze_label(phase)} (day of year)"

        fig, axes = plot_phase_comparison_map(
            static_field=clim_static,
            dynamic_field=clim_dynamic,
            lons=lons,
            lats=lats,
            label=label,
            title_prefix=f"{phase} ",
        )

        # You will set the actual caption/title in the paper; this just sets the filename.
        fig_num = 3 if phase == "FS" else 4  # adjust if you change ordering

        fig_name = format_fig_name(
            num=fig_num,
            short=f"climatology_{phase}_static_vs_dynamic_{YEAR_START}-{YEAR_END}",
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


if __name__ == "__main__":
    main()
