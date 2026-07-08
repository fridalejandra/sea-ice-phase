# ============================================================
# fig04-5_climatology_static_dynamic_maps.py  (UPDATED — min-N=10 display floor restored)
# ============================================================
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fig4-5_FS_MS_climatology_static_vs_dynamic.py

Compare static vs dynamic FS/MS climatologies:

  - FS: static, dynamic, dynamic − static
  - MS: static, dynamic, dynamic − static

Static files:
    /user/geog/falejandraperez/sea-ice-phase/results/SMMR_phase/FS_thr15_k5/FS_YYYY.nc
    /user/geog/falejandraperez/sea-ice-phase/results/SMMR_phase/MS_thr15_k5/MS_YYYY.nc

Dynamic files:
    /user/geog/falejandraperez/sea-ice-phase/data/SMMR_phase/dynamic/k5_q70/FS/FS_YYYY.nc
    /user/geog/falejandraperez/sea-ice-phase/data/SMMR_phase/dynamic/k5_q70/MS/MS_YYYY.nc

Notes on timing axes:
- FS is shown in CALENDAR day-of-year over Feb 15 (DOY 46) to Sep 30 (DOY 273).
- MS crosses the calendar year boundary, so it is remapped to a continuous axis:
    "days since Aug 15" (Aug 15 = 0; Feb 28 ~ 197).

Masking (2024 revision, final):
- Panels (a) static and (b) dynamic are each masked by a simple per-method
  DISPLAY MINIMUM: at least DISPLAY_MIN_YEARS (10) real valid years out of
  43, plus valid_ocean. This is a sample-size floor, not a comparison bar —
  its only job is to keep a climatological mean from being drawn off a
  handful of noisy years.
- A diagnostic sweep (thresholds 0%, 30%, 50%, 65%, 80% of years) showed
  that applying the strict 80%-of-years JOINT criterion (both methods
  reliable) to these single-method panels was wrong: it cut real ice-edge
  signal for a reason that has nothing to do with a single-method display
  (e.g. FS static p95 shifted by 33 days between no floor and 80%). It also
  showed that removing the floor entirely is worse, not better: the
  unmasked min/max value in every field (FS/MS, static/dynamic) was backed
  by exactly 1 of 43 years — pure single-year noise setting the color scale
  and the outer ring of the map.
- Panel (c), the difference, is NOT separately masked. dynamic - static is
  NaN wherever either input is NaN, so it automatically reduces to the
  joint (both-methods-reliable) footprint on its own.
- The strict 80%-of-years JOINT criterion (both methods reliable) remains
  correct and unchanged for Figs. 6-8, where an actual comparison between
  methods is being made — this floor is scoped only to this figure's
  single-method display panels.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import sys
from pathlib import Path
from glob import glob
import numpy as np
import xarray as xr

# ensure Ch2/ is on sys.path so "utils.*" imports work
HERE = Path(__file__).resolve().parent   # .../Ch2/figures
CH2_ROOT = HERE.parent                    # .../Ch2
if str(CH2_ROOT) not in sys.path:
    sys.path.insert(0, str(CH2_ROOT))


PROJECT_ROOT = Path("/user/geog/falejandraperez/sea-ice-phase")
from utils.ch2_fig_utils import (  # noqa: E402
    set_mpl_defaults,
    format_fig_name,
    get_fig_path,
    save_and_upload,
    plot_phase_comparison_map,
)

# ---------------------------------------------------------------------
# PATH CONFIG
# ---------------------------------------------------------------------

STATIC_ROOT = PROJECT_ROOT / "data" / "SMMR_phase" / "static"

DYN_FS_DIR = (
    PROJECT_ROOT
    / "data"
    / "SMMR_phase"
    / "dynamic"
    / "k5_q70"
    / "FS"
)

DYN_MS_DIR = (
    PROJECT_ROOT
    / "data"
    / "SMMR_phase"
    / "dynamic"
    / "k5_q70"
    / "MS"
)

SECTOR_FILE = PROJECT_ROOT / "data" / "canonical_sectors.nc"

REMOTE_ROOT = "gdrive:sea-ice-phase/results/Ch2_Figures"
SUBFOLDER = ""

PHASES = ["FS", "MS"]
YEAR_START = 1979
YEAR_END = 2024

# Display floor for panels (a)/(b) — a sample-size minimum, NOT the
# comparison-grade active80 criterion used in Figs. 6-8.
DISPLAY_MIN_YEARS = 10

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------


def load_phase_climatology(
    phase: str,
    mode: str,
    year_start: int,
    year_end: int,
) -> tuple[xr.DataArray, xr.DataArray, int]:
    """
    Load FS/MS files for given mode and return climatological mean over years.

    mode:
      - "static": results/SMMR_phase/<phase>_thr15_k5/<phase>_YYYY.nc
      - "dynamic": points at DYN_FS_DIR / DYN_MS_DIR above

    Assumes variable is named <phase> in each file.

    Returns
    -------
    clim : xr.DataArray
        Climatological mean over years.
    nvalid : xr.DataArray
        Per-pixel count of years with a finite (valid) detection.
    n_years : int
        Total number of years included (denominator, informational only
        now that the display floor is an absolute count rather than a
        fraction). Bad years already excluded upstream by
        compute_phase_dates_v2.py, so this should be 43 for 1979-2024.
    """
    if mode == "static":
        phase_dir = STATIC_ROOT / "thr15_k5" / phase
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
    nvalid = np.isfinite(da).sum("year").compute()
    return clim, nvalid, len(years_sel)


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


def load_ms_climatology_dsa(method: str, year_start: int, year_end: int) -> xr.DataArray:
    """
    Load MS climatology in "days since Aug 15" coordinate from results/anomalies.

    Expects variables written by the updated climatology/anomaly script:
      - MS_static_clim_dsa in results/anomalies/MS_static_climatology.nc
      - MS_dynamic_clim_dsa in results/anomalies/MS_dynamic_climatology.nc
    """
    suffix = "thr15_k5" if method == "static" else "k5_q70"
    clim_path = PROJECT_ROOT / "data" / "anomalies" / "SMMR" / f"MS_{method}_{suffix}_climatology.nc"
    if not clim_path.exists():
        raise FileNotFoundError(f"Missing MS climatology file: {clim_path}")

    ds = xr.open_dataset(clim_path, decode_times=False)
    var = f"MS_{method}_{suffix}_clim_dsa"
    if var not in ds:
        raise KeyError(
            f"{var} not found in {clim_path}. "
            "Did you rerun compute_FS_MS_anomalies_static_dynamic.py after adding the _dsa outputs?"
        )

    da = ds[var].load()
    ds.close()
    return da


def make_display_floor_mask(
    nvalid: xr.DataArray,
    valid_ocean: xr.DataArray,
    min_years: int = DISPLAY_MIN_YEARS,
) -> np.ndarray:
    """
    Per-method display floor: at least `min_years` real valid years,
    within valid_ocean. An absolute count, not a fraction of the record —
    this is deliberately NOT the same object as the active80 comparison
    criterion used in Figs. 6-8.

    Returns a plain numpy boolean array to avoid coordinate-alignment
    issues between calendar-DOY-based nvalid arrays and the separately
    loaded DSA climatology fields used for MS display.
    """
    nvalid_np = np.asarray(nvalid)
    ocean_np = np.asarray(valid_ocean)
    return (nvalid_np >= min_years) & ocean_np


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------


def main():
    set_mpl_defaults()

    ds_mask = xr.open_dataset(SECTOR_FILE)
    valid_ocean = ds_mask["valid_ocean"].astype(bool)
    ds_mask.close()

    for phase in PHASES:
        print(f"Processing climatology for {phase}")

        clim_static, nvalid_static, n_years_static = load_phase_climatology(phase, "static", YEAR_START, YEAR_END)
        clim_dynamic, nvalid_dynamic, n_years_dynamic = load_phase_climatology(phase, "dynamic", YEAR_START, YEAR_END)

        # --- phase-specific plotting coordinates ---
        if phase == "FS":
            # Feb 15–Sep 30 (calendar DOY)
            label = f"{freeze_label(phase)} (day of year)"
            field_vmin, field_vmax = 46, 273

        elif phase == "MS":
            # MS climatology should NOT be computed by averaging calendar DOY and then wrapping.
            # Instead, load the pre-wrapped climatology (days since Aug 15) produced by
            # compute_FS_MS_anomalies_static_dynamic.py: MS_*_clim_dsa
            clim_static = load_ms_climatology_dsa(method="static", year_start=YEAR_START, year_end=YEAR_END)
            clim_dynamic = load_ms_climatology_dsa(method="dynamic", year_start=YEAR_START, year_end=YEAR_END)

            label = f"{freeze_label(phase)} (days since Aug 15)"
            field_vmin, field_vmax = 0, 210

        else:
            label = f"{freeze_label(phase)}"
            field_vmin, field_vmax = None, None

        # Per-method display floor (min-N = 10 years) — replaces the
        # comparison-grade active80 bar that was wrongly applied to these
        # single-method panels. Panel (c) needs no separate mask: it
        # reduces to the joint footprint automatically via NaN propagation.
        static_reliable = make_display_floor_mask(nvalid_static, valid_ocean)
        dynamic_reliable = make_display_floor_mask(nvalid_dynamic, valid_ocean)

        clim_static = clim_static.where(xr.DataArray(static_reliable, dims=clim_static.dims))
        clim_dynamic = clim_dynamic.where(xr.DataArray(dynamic_reliable, dims=clim_dynamic.dims))

        n_ocean = int(np.asarray(valid_ocean).sum())
        n_joint = int((static_reliable & dynamic_reliable).sum())
        print(f"  [display floor >= {DISPLAY_MIN_YEARS} yrs] {phase}: "
              f"static={int(static_reliable.sum())} (of {n_ocean} valid-ocean), "
              f"dynamic={int(dynamic_reliable.sum())} (of {n_ocean} valid-ocean), "
              f"joint (what panel c will show)={n_joint}")

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
        fig_num = 4 if phase == "FS" else 5  # adjust if you change ordering

        fig_name = format_fig_name(
            num=fig_num,
            short=f"climatology_{phase}_static_vs_dynamic_{YEAR_START}-{YEAR_END}_minN{DISPLAY_MIN_YEARS}",
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
            remote_subdir="",
        )


if __name__ == "__main__":
    main()