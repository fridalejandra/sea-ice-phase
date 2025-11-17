#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from glob import glob
import subprocess

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import Normalize
from matplotlib.path import Path as MplPath

# ---------------------------------------------------------------------
# STANDARD FIGURE SIZES for Word/Google Docs
# ---------------------------------------------------------------------
FIGSIZE_SINGLE = (5, 5)        # single map/plot
FIGSIZE_DOUBLE = (8, 4)        # 2 panels side-by-side
FIGSIZE_TRIPLE = (10.5, 4)     # 3 panels side-by-side
FIGSIZE_WIDE   = (11, 4)       # extra-wide triple if needed
FIGSIZE_TALL   = (5, 7)        # for vertical stack if ever used
# ---------------------------------------------------------------------
DPI = 300

# =======================
# CONFIG
# =======================

# --- choose method + paths --- #
MODE = "dynamic_quantile"      # "static" or "dynamic_quantile"

YEAR_START = 1979
YEAR_END   = 2024

# Static FS/MS/ME (thr=0.15, k=5)
STATIC_ROOT = "/user/geog/falejandraperez/sea-ice-phase/results/SMMR_phase"

# Dynamic FS/MS/ME (quantile, k=5, p=0.70)
DYN_ROOT    = "/user/geog/falejandraperez/sea-ice-phase/results/static_v2_slopeH/dynamic"
DYN_SCHEME  = "quantile_k5"   # folder created by your dynamic script
DYN_TAG     = "p0.7"          # subfolder for parameters ("p0.7", "alpha1.0", etc.)

OUT_DIR = "/user/geog/falejandraperez/sea-ice-phase/results/Ch2_Figures/climatology"
os.makedirs(OUT_DIR, exist_ok=True)

PHASES = ["FS", "MS", "ME"]   # all phases

# rclone remote (EDIT THIS to match your config)
# Example: "gdrive:sea-ice-phase/Ch2_Figures/climatology"
RCLONE_REMOTE = "gdrive:sea-ice-phase/Ch2_Figures/climatology"


# =======================
# HELPERS
# =======================

def make_polar_axes(ax):
    """
    Configure an existing axes as south polar stereographic with circular boundary.
    This avoids creating new axes on top of the subplots.
    """
    proj = ccrs.SouthPolarStereo()
    ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())

    # circular boundary
    theta = np.linspace(0, 2 * np.pi, 200)
    center = np.array([0.5, 0.5])
    radius = 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T * radius + center
    circle = MplPath(verts)
    ax.set_boundary(circle, transform=ax.transAxes)

    # land/ocean + coastlines
    ax.add_feature(cfeature.OCEAN, facecolor="black", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="0.7", edgecolor="0.7", zorder=1)
    ax.coastlines(linewidth=0.3, zorder=2)

    ax.gridlines(draw_labels=False, linewidth=0.3, color="0.5",
                 alpha=0.5, linestyle="--")
    return ax


def load_phase_climatology(phase, mode, year_start, year_end):
    """
    Load FS/MS/ME files for given mode and return mean over years.
    Assumes each file is phase_YEAR.nc with variable named phase.
    """
    if mode == "static":
        phase_dir = os.path.join(STATIC_ROOT, f"{phase}_thr15_k5")
        pattern = os.path.join(phase_dir, f"{phase}_*.nc")
    elif mode == "dynamic_quantile":
        # e.g. /.../dynamic/quantile_k5/FS/p0.7/FS_YYYY.nc
        phase_dir = os.path.join(DYN_ROOT, DYN_SCHEME, phase, DYN_TAG)
        pattern = os.path.join(phase_dir, f"{phase}_*.nc")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    files = sorted(glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found for {phase} with pattern {pattern}")

    # extract years from filenames
    years = []
    for f in files:
        base = os.path.basename(f)
        try:
            y = int(base.split("_")[1].split(".")[0])
            years.append(y)
        except Exception:
            continue

    years = np.array(years)
    mask = (years >= year_start) & (years <= year_end)
    if not mask.any():
        raise ValueError(f"No years in [{year_start}, {year_end}] for {phase}")

    files_sel = [f for f, m in zip(files, mask) if m]
    years_sel = years[mask]

    ds = xr.open_mfdataset(files_sel, combine="nested", concat_dim="year")
    ds = ds.assign_coords(year=("year", years_sel))

    da = ds[phase]  # (year, y, x)
    clim = da.mean("year", skipna=True)  # (y, x)

    return clim


def upload_with_rclone(local_path, remote_base):
    """Copy a single file to the given rclone remote path."""
    if not RCLONE_REMOTE:
        return
    fname = os.path.basename(local_path)
    remote_path = f"{remote_base}/{fname}"
    cmd = ["rclone", "copyto", local_path, remote_path]
    try:
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        print(f"Uploaded via rclone to: {remote_path}")
    except Exception as e:
        print(f"rclone upload failed for {fname}: {e}")


# =======================
# MAIN PLOT
# =======================

def make_figure(shared_colorbar: bool):
    # Load climatologies per phase
    clim = {}
    for ph in PHASES:
        print(f"Loading climatology for {ph} [{MODE}]...")
        clim[ph] = load_phase_climatology(ph, MODE, YEAR_START, YEAR_END)

    sample = clim[PHASES[0]]
    x = sample["x"]
    y = sample["y"]

    # DOY colormap + ticks
    cmap = plt.cm.twilight_shifted
    # full-year range so FS/MS/ME are comparable
    norm = Normalize(vmin=0, vmax=365)
    ticks = [15, 105, 196, 288, 365]
    labels = ["Jan", "Apr", "Jul", "Oct", "Dec"]

    fig = plt.figure(figsize=FIGSIZE_TRIPLE, dpi=DPI)
    axes = []
    meshes = []

    for i, ph in enumerate(PHASES):
        ax = fig.add_subplot(1, len(PHASES), i + 1,
                             projection=ccrs.SouthPolarStereo())
        ax = make_polar_axes(ax)

        da = clim[ph]
        mesh = ax.pcolormesh(
            x, y, da,
            transform=ccrs.SouthPolarStereo(),
            cmap=cmap,
            norm=norm
        )

        if ph == "FS":
            ttl = "Freeze start (FS)"
        elif ph == "MS":
            ttl = "Melt start (MS)"
        else:
            ttl = "Melt end (ME)"

        ax.set_title(ttl, fontsize=9)
        axes.append(ax)
        meshes.append(mesh)

        if not shared_colorbar:
            # individual colorbar per panel
            cb = fig.colorbar(mesh, ax=ax,
                              orientation="horizontal",
                              fraction=0.046, pad=0.04)
            cb.set_ticks(ticks)
            cb.set_ticklabels(labels)
            cb.ax.tick_params(labelsize=7)
            cb.outline.set_visible(False)

    if shared_colorbar:
        # Shared colorbar along bottom
        cax = fig.add_axes([0.15, 0.08, 0.7, 0.03])
        cb = fig.colorbar(meshes[0], cax=cax, orientation="horizontal")
        cb.set_ticks(ticks)
        cb.set_ticklabels(labels)
        cb.ax.tick_params(labelsize=8)
        cb.outline.set_visible(False)
        cb.set_label("Day of year", fontsize=9, labelpad=3)

    fig.suptitle(
        f"{MODE} phase climatology ({YEAR_START}–{YEAR_END})",
        fontsize=11
    )

    # Output naming depends on colorbar mode
    cbar_tag = "sharedcbar" if shared_colorbar else "separatecbars"
    out_name = f"fig_phase_climatology_{MODE}_{YEAR_START}_{YEAR_END}_{cbar_tag}.png"
    save_path = os.path.join(OUT_DIR, out_name)

    # layout
    if shared_colorbar:
        plt.tight_layout(rect=[0, 0.14, 1, 0.94])
    else:
        plt.tight_layout(rect=[0, 0.06, 1, 0.94])

    plt.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved locally: {save_path}")

    # rclone upload
    upload_with_rclone(save_path, RCLONE_REMOTE)


def main():
    # 1) Shared colorbar across all phases
    make_figure(shared_colorbar=True)
    # 2) Individual colorbars per phase
    make_figure(shared_colorbar=False)


if __name__ == "__main__":
    main()
