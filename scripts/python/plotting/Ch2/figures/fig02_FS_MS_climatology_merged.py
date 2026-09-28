#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================
# fig02_FS_MS_climatology_merged.py
# ============================================================
"""
Merged FS + MS climatology: single 2x3 figure (Fig. 2).

Top row (FS): static, dynamic, dynamic - static
Bottom row (MS): static, dynamic, dynamic - static

Reuses all logic from fig04-5_climatology_static_dynamic_maps.py
"""

import sys
from pathlib import Path
from glob import glob

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.python.plotting.Ch2.utils.ch2_fig_utils import (
    set_mpl_defaults,
    format_fig_name,
    get_fig_path,
    save_and_upload,
)

# ---- Imports for plotting utilities ----
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.path import Path as MplPath

# ---- PATH CONFIG ----
STATIC_ROOT = PROJECT_ROOT / "data" / "SMMR_phase" / "static"
DYN_FS_DIR = PROJECT_ROOT / "data" / "SMMR_phase" / "dynamic" / "k5_q70" / "FS"
DYN_MS_DIR = PROJECT_ROOT / "data" / "SMMR_phase" / "dynamic" / "k5_q70" / "MS"
SECTOR_FILE = PROJECT_ROOT / "data" / "canonical_sectors.nc"

REMOTE_ROOT = "gdrive:sea-ice-phase/results/Ch2_Figures"
YEAR_START = 1979
YEAR_END = 2024
DISPLAY_MIN_YEARS = 10

# ---- HELPERS (copied from fig04-5_climatology_static_dynamic_maps.py) ----

def load_phase_climatology(phase, mode, year_start, year_end):
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
        raise FileNotFoundError(f"No files found for phase={phase}, mode={mode}, pattern={pattern}")

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
        raise ValueError(f"No years in [{year_start}, {year_end}] for phase={phase}, mode={mode}")

    files_sel = [f for f, m in zip(files, mask) if m]
    years_sel = years[mask]

    ds = xr.open_mfdataset(files_sel, combine="nested", concat_dim="year")
    ds = ds.assign_coords(year=("year", years_sel))
    da = ds[phase]
    clim = da.mean("year", skipna=True)
    nvalid = np.isfinite(da).sum("year").compute()
    return clim, nvalid, len(years_sel)


def load_ms_climatology_dsa(method, year_start, year_end):
    suffix = "thr15_k5" if method == "static" else "k5_q70"
    clim_path = PROJECT_ROOT / "data" / "anomalies" / "SMMR" / f"MS_{method}_{suffix}_climatology.nc"
    if not clim_path.exists():
        raise FileNotFoundError(f"Missing MS climatology file: {clim_path}")

    ds = xr.open_dataset(clim_path, decode_times=False)
    var = f"MS_{method}_{suffix}_clim_dsa"
    if var not in ds:
        raise KeyError(f"{var} not found in {clim_path}")

    da = ds[var].load()
    ds.close()
    return da


def make_display_floor_mask(nvalid, valid_ocean, min_years=DISPLAY_MIN_YEARS):
    nvalid_np = np.asarray(nvalid)
    ocean_np = np.asarray(valid_ocean)
    return (nvalid_np >= min_years) & ocean_np


def quick_stats(name, da):
    v = da.values
    v = v[np.isfinite(v)]
    if v.size == 0:
        print(f"{name}: all-NaN")
        return
    print(
        f"{name}: min={np.nanmin(v):.1f}, max={np.nanmax(v):.1f}, "
        f"p5={np.nanpercentile(v, 5):.1f}, p95={np.nanpercentile(v, 95):.1f}"
    )


# ---- PLOTTING FUNCTION ----

def plot_polar_field(ax, field, lons=None, lats=None, x=None, y=None, 
                     vmin=None, vmax=None, cmap='viridis', label=""):
    """
    Plot a single field on a polar stereographic axis.
    """
    proj = ccrs.SouthPolarStereo()
    
    if lons is not None and lats is not None:
        im = ax.pcolormesh(
            lons, lats, field,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            shading='auto',
            vmin=vmin,
            vmax=vmax,
        )
    elif x is not None and y is not None:
        im = ax.pcolormesh(
            x, y, field,
            transform=proj,
            cmap=cmap,
            shading='auto',
            vmin=vmin,
            vmax=vmax,
        )
    else:
        raise ValueError("Need either lons/lats or x/y coordinates")

    ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="0.8", edgecolor="0.6", zorder=1)
    ax.coastlines(linewidth=0.4, zorder=2)
    ax.gridlines(draw_labels=False, linewidth=0.3, color="0.5", alpha=0.5, linestyle="--")

    return im


# ---- MAIN ----

def main():
    set_mpl_defaults()

    ds_mask = xr.open_dataset(SECTOR_FILE)
    valid_ocean = ds_mask["valid_ocean"].astype(bool)
    ds_mask.close()

    # Create 2x3 figure (half-page size)
    proj = ccrs.SouthPolarStereo()
    fig = plt.figure(figsize=(5.5, 4), dpi=300)
    gs = fig.add_gridspec(2, 3, hspace=0.30, wspace=0.20)

    phases = ["FS", "MS"]
    colmaps = {"static": "viridis", "dynamic": "viridis", "diff": "RdBu_r"}
    
    for row_idx, phase in enumerate(phases):
        print(f"Loading climatology for {phase}")

        clim_static, nvalid_static, _ = load_phase_climatology(phase, "static", YEAR_START, YEAR_END)
        clim_dynamic, nvalid_dynamic, _ = load_phase_climatology(phase, "dynamic", YEAR_START, YEAR_END)

        # Phase-specific coordinates
        if phase == "FS":
            label = "Freeze start (day of year)"
            field_vmin, field_vmax = 46, 273
        elif phase == "MS":
            clim_static = load_ms_climatology_dsa("static", YEAR_START, YEAR_END)
            clim_dynamic = load_ms_climatology_dsa("dynamic", YEAR_START, YEAR_END)
            label = "Melt start (days since Aug 15)"
            field_vmin, field_vmax = 0, 210

        # Apply display floor
        static_reliable = make_display_floor_mask(nvalid_static, valid_ocean)
        dynamic_reliable = make_display_floor_mask(nvalid_dynamic, valid_ocean)

        clim_static = clim_static.where(xr.DataArray(static_reliable, dims=clim_static.dims))
        clim_dynamic = clim_dynamic.where(xr.DataArray(dynamic_reliable, dims=clim_dynamic.dims))

        n_ocean = int(np.asarray(valid_ocean).sum())
        n_joint = int((static_reliable & dynamic_reliable).sum())
        print(f"  [{phase} display floor >= {DISPLAY_MIN_YEARS} yrs] "
              f"static={int(static_reliable.sum())}, dynamic={int(dynamic_reliable.sum())}, joint={n_joint}")

        quick_stats(f"{phase} static", clim_static)
        quick_stats(f"{phase} dynamic", clim_dynamic)

        # Compute difference
        diff = clim_dynamic - clim_static

        # Extract coordinates for plotting
        if "lon" in clim_static.coords and "lat" in clim_static.coords:
            lons = clim_static["lon"]
            lats = clim_static["lat"]
            x, y = None, None
        elif "x" in clim_static.coords and "y" in clim_static.coords:
            x = clim_static["x"]
            y = clim_static["y"]
            lons, lats = None, None
        else:
            # Try to infer from coords
            coords_set = set(clim_static.coords)
            if {"lon", "lat"} <= coords_set:
                lons, lats = clim_static["lon"], clim_static["lat"]
                x, y = None, None
            else:
                x, y = clim_static["x"], clim_static["y"]
                lons, lats = None, None

        # Plot 3 panels for this phase
        for col_idx, (name, field, vmin, vmax, cmap) in enumerate([
            ("Static", clim_static, field_vmin, field_vmax, "viridis"),
            ("Dynamic", clim_dynamic, field_vmin, field_vmax, "viridis"),
            ("Difference", diff, -20, 20, "RdBu_r"),
        ]):
            ax = fig.add_subplot(gs[row_idx, col_idx], projection=proj)

            im = plot_polar_field(
                ax, field,
                lons=lons, lats=lats, x=x, y=y,
                vmin=vmin, vmax=vmax, cmap=cmap, label=label
            )

            # Title (reduced fontsize)
            ax.set_title(f"{phase} {name}", fontsize=8, fontweight="bold")

            # Colorbar (reduced fontsize)
            cbar = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.7)
            if name == "Difference":
                cbar.set_label("Days", fontsize=6)
            else:
                cbar.set_label(label, fontsize=6)
            cbar.ax.tick_params(labelsize=5)
            cbar.outline.set_visible(False)

            # Panel letter (reduced fontsize)
            letters = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
            letter_idx = row_idx * 3 + col_idx
            ax.text(0.02, 0.98, letters[letter_idx], transform=ax.transAxes,
                   ha="left", va="top", fontsize=9, fontweight="bold")

    fig_name = format_fig_name(
        num=2,
        short=f"FS_MS_climatology_static_vs_dynamic_{YEAR_START}-{YEAR_END}_minN{DISPLAY_MIN_YEARS}",
    )

    out_path = get_fig_path(
        project_root=PROJECT_ROOT,
        subfolder="",
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
