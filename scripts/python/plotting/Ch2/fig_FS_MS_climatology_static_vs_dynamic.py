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

Dynamic files:
    /user/geog/falejandraperez/sea-ice-phase/results/static_v2_slopeH/dynamic/quantile_k5/FS/p0.7/FS_YYYY.nc
    /user/geog/falejandraperez/sea-ice-phase/results/static_v2_slopeH/dynamic/quantile_k5/MS/p0.7/MS_YYYY.nc

Notes on timing axes:
- FS is shown in CALENDAR day-of-year over Feb 15 (DOY 46) to Sep 30 (DOY 273).
- MS crosses the calendar year boundary, so it is remapped to a continuous axis:
    "days since Aug 15" (Aug 15 = 0; Feb 28 ~ 197).
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

STATIC_ROOT = PROJECT_ROOT / "results" / "SMMR_phase"

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
SUBFOLDER = "climatology"

PHASES = ["FS", "MS"]
YEAR_START = 1979
YEAR_END = 2023

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
      - "dynamic": points at DYN_FS_DIR / DYN_MS_DIR above

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


def wrap_ms_to_aug15(da: xr.DataArray, aug15_doy: int = 227) -> xr.DataArray:
    """
    Remap MS calendar day-of-year to a continuous axis starting Aug 15.

    Any DOY < aug15 is assumed to occur in the *following* calendar year.
    We add 365 so Aug..Feb becomes continuous:
      [227..365] U [366..(365+59)].
    """
    return xr.where(da < aug15_doy, da + 365, da)


def ms_days_since_aug15(da_wrapped: xr.DataArray, aug15_doy: int = 227) -> xr.DataArray:
    """Convert wrapped MS DOY to 'days since Aug 15' (Aug 15 -> 0)."""
    return da_wrapped - aug15_doy


def freeze_label(phase: str) -> str:
    if phase == "FS":
        return "Freeze start"
    elif phase == "MS":
        return "Melt start"
    return phase


def quick_stats(name: str, da: xr.DataArray) -> None:
    v = da.values
    v = v[np.isfinite(v)]
    if v.size == 0:
        print(f"{name}: all-NaN")
        return
    print(
        f"{name}: min={np.nanmin(v):.1f}, max={np.nanmax(v):.1f}, "
        f"p5={np.nanpercentile(v, 5):.1f}, p95={np.nanpercentile(v, 95):.1f}"
    )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------


def main():
    set_mpl_defaults()

    for phase in PHASES:
        print(f"Processing climatology for {phase}")

        clim_static = load_phase_climatology(phase, "static", YEAR_START, YEAR_END)
        clim_dynamic = load_phase_climatology(phase, "dynamic", YEAR_START, YEAR_END)

        # --- phase-specific plotting coordinates ---
        if phase == "FS":
            # Feb 15–Sep 30 (calendar DOY)
            label = f"{freeze_label(phase)} (day of year)"
            field_vmin, field_vmax = 46, 273
        elif phase == "MS":
            # Aug 15–Feb 28 crosses calendar year:
            # remap to continuous axis: days since Aug 15 (Aug 15 = 0)
            clim_static = ms_days_since_aug15(wrap_ms_to_aug15(clim_static))
            clim_dynamic = ms_days_since_aug15(wrap_ms_to_aug15(clim_dynamic))

            label = f"{freeze_label(phase)} (days since Aug 15)"
            # Aug 15 -> 0; Feb 28 ~ 197 (non-leap). Give a little headroom.
            field_vmin, field_vmax = 0, 210
        else:
            label = f"{freeze_label(phase)}"
            field_vmin, field_vmax = None, None

        # Optional sanity printout (kept on by default — comment out if annoying)
        quick_stats(f"{phase} static", clim_static)
        quick_stats(f"{phase} dynamic", clim_dynamic)

        fig, axes = plot_phase_comparison_map(
            static_field=clim_static,
            dynamic_field=clim_dynamic,
            label=label,
            title_prefix=f"{phase} ",
            diff_vlim=20,
            field_vmin=field_vmin,
            field_vmax=field_vmax,
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
