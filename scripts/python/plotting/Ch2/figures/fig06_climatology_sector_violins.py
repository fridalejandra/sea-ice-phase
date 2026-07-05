#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Walsh-style violins: static vs dynamic FS/MS climatological timing by sector.

Static:
  data/SMMR_phase/static/thr15_k5/{FS,MS}/{FS,MS}_YYYY.nc
    - variable FS or MS (calendar DOY)

Dynamic:
  data/SMMR_phase/dynamic/k5_q70/{FS,MS}/{FS,MS}_YYYY.nc
    - variable FS or MS (calendar DOY)

Important:
- FS is fine to average in calendar DOY.
- MS crosses the year boundary (Aug–Feb), so we convert EACH YEAR to a continuous
  axis "days since Aug 15" BEFORE computing climatologies and violins.
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

from utils.plot_utils import (  # noqa: E402
    set_mpl_defaults,
    get_fig_path,
    save_and_upload,
    PROJECT_ROOT_CLUSTER,
)

set_mpl_defaults()
sns.set_style("whitegrid")

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
YEAR_MIN = 1979
YEAR_MAX = 2024

STATIC_DIR = PROJECT_ROOT_CLUSTER / "data" / "SMMR_phase" / "static"
DYN_ROOT = (
    PROJECT_ROOT_CLUSTER
    / "data"
    / "SMMR_phase"
    / "dynamic"
    / "k5_q70"
)

DYN_DIR_FS = DYN_ROOT / "FS"
DYN_DIR_MS = DYN_ROOT / "MS"

SECTOR_FILE = PROJECT_ROOT_CLUSTER / "data" / "canonical_sectors.nc"

# numeric IDs used internally; labels only used for tick text
sector_labels = {
    1: "Amundsen–\nBellingshausen",
    2: "Weddell",
    3: "King Haakon VII",
    4: "East Antarctica",
    5: "Ross–\nAmundsen",
}
sector_ids = [1, 2, 3, 4, 5]

AUG15_DOY = 227  # Aug 15 (non-leap)

MIN_FRAC_ACTIVE = 0.80  # same active-pixel criterion as fig07 trends

# ---------------------------------------------------------------------
# Sector mask
# ---------------------------------------------------------------------
ds_sect = xr.open_dataset(SECTOR_FILE)
sector_mask = ds_sect["sector_id"].values
ocean_mask = ds_sect["valid_ocean"].astype(bool).values
ds_sect.close()

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def ms_to_days_since_aug15(ms_da: xr.DataArray, aug15_doy: int = AUG15_DOY) -> xr.DataArray:
    """
    Convert MS calendar DOY -> continuous days since Aug 15.
    Aug 15 -> 0; Feb 28 ~ 197. NaNs preserved.
    """
    wrapped = xr.where(ms_da < aug15_doy, ms_da + 365, ms_da)
    return wrapped - aug15_doy


def _load_static_year(phase: str, year: int) -> xr.DataArray | None:
    """
    phase: 'FS' or 'MS'
    static file: data/SMMR_phase/static/thr15_k5/<phase>/<phase>_YYYY.nc
    variable: <phase> (y, x)
    """
    fpath = STATIC_DIR / "thr15_k5" / phase / f"{phase}_{year}.nc"
    if not fpath.exists():
        return None

    ds = xr.open_dataset(fpath)

    if phase not in ds:
        ds.close()
        return None

    da = ds[phase].load()
    ds.close()

    if not np.any(np.isfinite(da.values)):
        return None

    return da


def _load_dynamic_year(phase: str, year: int) -> xr.DataArray | None:
    """
    Dynamic FS/MS files:
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
        ds.close()
        raise KeyError(f"{phase} not in {fpath}; vars={list(ds.data_vars)}")

    da = ds[phase].load()
    ds.close()

    if not np.any(np.isfinite(da.values)):
        return None

    return da


def compute_climatologies_for_phase(phase: str) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Compute static and dynamic climatologies for one phase (FS or MS),
    using ONLY years where both exist.

    For MS: convert EACH YEAR to 'days since Aug 15' BEFORE averaging.
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

    stat_list = []
    dyn_list = []

    for y in common_years:
        s = static_dict[y]
        d = dyn_dict[y]

        # Critical: MS wrap per-year (avoid calendar-year mean artifact)
        if phase == "MS":
            s = ms_to_days_since_aug15(s)
            d = ms_to_days_since_aug15(d)

        stat_list.append(s.expand_dims(year=[y]))
        dyn_list.append(d.expand_dims(year=[y]))

    stat_all = xr.concat(stat_list, dim="year")
    dyn_all = xr.concat(dyn_list, dim="year")

    stat_clim = stat_all.mean("year", skipna=True)
    dyn_clim = dyn_all.mean("year", skipna=True)

    # Active-pixel mask: valid timing in >= MIN_FRAC_ACTIVE of years for
    # BOTH methods (identical criterion to fig07 MIN_FRAC_ACTIVE)
    n_years = len(common_years)
    frac_stat = np.isfinite(stat_all).sum("year") / n_years
    frac_dyn = np.isfinite(dyn_all).sum("year") / n_years
    active_mask = (frac_stat >= MIN_FRAC_ACTIVE) & (frac_dyn >= MIN_FRAC_ACTIVE)

    return stat_clim, dyn_clim, active_mask


# ---------------------------------------------------------------------
# Compute climatologies
# ---------------------------------------------------------------------
print("Computing FS static/dynamic climatologies...")
fs_stat_clim, fs_dyn_clim, fs_active = compute_climatologies_for_phase("FS")

print("Computing MS static/dynamic climatologies (days since Aug 15)...")
ms_stat_clim, ms_dyn_clim, ms_active = compute_climatologies_for_phase("MS")

print(f"[INFO] Active mask @ {MIN_FRAC_ACTIVE:.2f}: "
      f"FS={int(fs_active.values.sum())}, MS={int(ms_active.values.sum())}")

# Check grid matches sector mask
if fs_stat_clim.shape != sector_mask.shape:
    raise ValueError(
        f"FS climatology shape {fs_stat_clim.shape} "
        f"does not match sector mask {sector_mask.shape}"
    )

# ---------------------------------------------------------------------
# Build + plot violins (run for both masking variants)
# ---------------------------------------------------------------------
def build_and_plot_violins(variant: str) -> None:
    if variant == "active80":
        fs_stat_clim_v = fs_stat_clim.where(fs_active)
        fs_dyn_clim_v = fs_dyn_clim.where(fs_active)
        ms_stat_clim_v = ms_stat_clim.where(ms_active)
        ms_dyn_clim_v = ms_dyn_clim.where(ms_active)
    else:
        fs_stat_clim_v = fs_stat_clim
        fs_dyn_clim_v = fs_dyn_clim
        ms_stat_clim_v = ms_stat_clim
        ms_dyn_clim_v = ms_dyn_clim

    records: list[dict] = []


    def add_phase_records(
        phase_name: str,
        da_static: xr.DataArray,
        da_dynamic: xr.DataArray,
        ylim_min: float,
        ylim_max: float,
        value_label: str,
    ) -> None:
        """
        Extract static/dynamic climatology values by sector into records.
        value_label: name of the y variable ('doy' for FS, 'dsa' for MS).
        """
        for sec in sector_ids:
            mask = (sector_mask == sec) & ocean_mask

            stat_vals = da_static.where(mask).values.ravel()
            dyn_vals = da_dynamic.where(mask).values.ravel()

            stat_vals = stat_vals[np.isfinite(stat_vals)]
            dyn_vals = dyn_vals[np.isfinite(dyn_vals)]

            # Clip to tidy plotting range
            stat_vals = stat_vals[(stat_vals >= ylim_min) & (stat_vals <= ylim_max)]
            dyn_vals = dyn_vals[(dyn_vals >= ylim_min) & (dyn_vals <= ylim_max)]

            for v in stat_vals:
                records.append(
                    {"phase": phase_name, "sector": sec, "method": "Static", value_label: float(v)}
                )
            for v in dyn_vals:
                records.append(
                    {"phase": phase_name, "sector": sec, "method": "Dynamic", value_label: float(v)}
                )


    # FS: calendar DOY (choose a sane plotting window)
    add_phase_records("FS", fs_stat_clim_v, fs_dyn_clim_v, 46, 273, value_label="value")

    # MS: days since Aug 15 (0..~197) — give headroom
    add_phase_records("MS", ms_stat_clim_v, ms_dyn_clim_v, 0, 210, value_label="value")

    df = pd.DataFrame.from_records(records)

    # ---------------------------------------------------------------------
    # Plot violins
    # ---------------------------------------------------------------------
    sns.set(style="whitegrid")

    palette = {"Static": "#2166ac", "Dynamic": "#d97a00"}
    sector_order = sector_ids

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, dpi=300)

    titles = {"FS": "(a) FS", "MS": "(b) MS"}

    for ax, phase_name in zip(axes, ["FS", "MS"]):
        sub = df[df["phase"] == phase_name]

        # Robust outline violins: side-by-side (avoids split+fill=False issues)
        sns.violinplot(
            data=sub,
            x="sector",
            y="value",
            hue="method",
            order=sector_order,
            palette=palette,
            dodge=True,        # side-by-side
            split=False,
            inner="quartile",
            linewidth=1.8,
            cut=0,
            ax=ax,
            fill=False,
        )

        ax.set_xticklabels([sector_labels[i] for i in sector_order])
        ax.set_title(titles[phase_name], fontweight="bold", pad=8)

        if phase_name == "FS":
            ax.set_ylabel("Freeze start (day of year)")
            ax.set_ylim(46, 273)
        else:
            ax.set_ylabel("Melt start (days since Aug 15)")
            ax.set_ylim(0, 210)

        ax.tick_params(axis="x", rotation=0)

    # Legend once, outside
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(
        handles,
        labels,
        title="",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
    )
    axes[1].get_legend().remove()
    axes[1].set_xlabel("Sector")

    fig.subplots_adjust(right=0.82)

    # ---------------------------------------------------------------------
    # Save / upload
    # ---------------------------------------------------------------------
    out_path = get_fig_path(
        PROJECT_ROOT_CLUSTER,
        subfolder="",
        fig_name=f"Fig06_FS_MS_climatology_static_vs_dynamic_violins_{variant}.png",
    )

    save_and_upload(
        fig,
        out_path,
        remote_root="gdrive:sea-ice-phase/results/Ch2_Figures",
        remote_subdir="",
    )

# ---------------------------------------------------------------------
# Run both variants
# ---------------------------------------------------------------------
for _variant in ["unmasked", "active80"]:
    print(f"Building violins: {_variant}")
    build_and_plot_violins(_variant)
