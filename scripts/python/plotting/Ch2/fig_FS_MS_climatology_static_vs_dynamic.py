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

Dynamic files (percentile p=0.7, k=5) – NEW LOCATION:
    /user/geog/falejandraperez/sea-ice-phase/results/dynamic_thresholds/FS/FS_YYYY.nc
    /user/geog/falejandraperez/sea-ice-phase/results/dynamic_thresholds/MS/MS_YYYY.nc
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
# PATH CONFIG
# ---------------------------------------------------------------------

# Static FS/MS (old slope+15% method)
STATIC_ROOT = PROJECT_ROOT / "results" / "SMMR_phase"

# Dynamic FS/MS (percentile p=0.7, k=5) – actual existing outputs
DYN_FS_DIR = (
    PROJECT_ROOT
    / "results"
    / "static_v2_slopeH"
    / "dynamic"
    / "quantile_k5"
    / "FS"
    / "p0.7"
)

DYN_MS_DIR = (
    PROJECT_ROOT
    / "results"
    / "static_v2_slopeH"
    / "dynamic"
    / "quantile_k5"
    / "MS"
    / "p0.7"
)


REMOTE_ROOT = "gdrive:sea-ice-phase/Results/Ch2_Figures"
SUBFOLDER   = "climatology"

PHASES = ["FS", "MS"]
YEAR_START = 1979
YEAR_END   = 2023

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def load_phase_climatology(
    phase: str,
    mode: str,
    year_start: int,
    year_end: int,
) -> xr.DataArray:
    """
    Load FS/MS files for given mode and return climatological mean over years.

    mode:
      - "static": results/SMMR_phase/<phase>_thr15_k5/<phase>_YYYY.nc
      - "dynamic": results/dynamic_thresholds/<phase>/<phase>_YYYY.nc

    Assumes variable is named <phase> in each file.
    """
    if mode == "static":
        phase_dir = STATIC_ROOT / f"{phase}_thr15_k5"
    elif mode == "dynamic":
        if phase == "FS":
            phase_dir = DYN_FS_DIR
        elif phase == "MS":
            phase_dir = DYN_MS_DIR
        else:
            raise ValueError(f"Unknown phase for dynamic: {phase}")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    pattern = str(phase_dir / f"{phase}_*.nc")
    files = sorted(glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No files found for phase={phase}, mode={mode}, pattern={pattern}"
        )

    # infer years from filenames like "FS_1979.nc"
    years = []
    for f in files:
        base = Path(f).name
        try:
            y = int(base.split("_")[1].split(".")[0])
            years.append(y)
        except Exception:
            continue

    years = np.array(years)
    mask = (years >= year_start) & (years <= year_end)
    if not mask.any():
        raise ValueError(
            f"No years in [{year_start}, {year_end}] for phase={phase}, mode={mode}"
        )

    files_sel = [f for f, m in zip(files, mask) if m]
    years_sel = years[mask]

    ds = xr.open_mfdataset(files_sel, combine="nested", concat_dim="year")
    ds = ds.assign_coords(year=("year", years_sel))

    da = ds[phase]
    clim = da.mean("year", skipna=True)

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

    # Use any static FS file to get grid and coords
    example_file = STATIC_ROOT / "FS_thr15_k5" / "FS_1979.nc"
    example = xr.open_dataset(example_file)

    # Prefer lon/lat if they exist; otherwise fall back to x/y
    if {"lon", "lat"} <= set(example.coords):
        lons = example["lon"]
        lats = example["lat"]
    else:
        # fall back: treat x,y as "lon,lat" for plotting purposes
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

        # You’ll title/caption in the paper; this is just the filename index.
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
