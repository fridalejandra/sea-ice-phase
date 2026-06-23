#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ch2_fig_utils.py

Shared plotting + I/O utilities for Chapter 2 figures
in the sea-ice-phase project.

All figure scripts under scripts/python/plotting/Ch2
should import from here instead of re-implementing
their own map, naming, or rclone logic.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import Iterable, Dict, Any, Optional

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.path import Path as MplPath
import seaborn as sns


# ---------------------------------------------------------------------
# GLOBAL STYLE
# ---------------------------------------------------------------------

DPI = 300

FIGSIZE_SINGLE = (5, 5)
FIGSIZE_DOUBLE = (8, 4)
FIGSIZE_TRIPLE = (10.5, 4)
FIGSIZE_WIDE   = (11, 4)
FIGSIZE_TALL   = (5, 7)


def set_mpl_defaults() -> None:
    """Set a consistent style for all Ch2 figures."""
    plt.rcParams.update({
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.autolayout": False,
    })
    sns.set_theme(style="whitegrid", font_scale=0.9)
from pathlib import Path
import xarray as xr
import numpy as np
# ... (rest of imports already there)


# ---------------------------------------------------------------------
# PATH CONSTANTS  (EDIT ONLY THESE IF YOU MOVE THINGS)
# ---------------------------------------------------------------------

PROJECT_ROOT_CLUSTER = Path("/user/geog/falejandraperez/sea-ice-phase")

# Static (old) phase, with window sensitivities (3,5,7 day)
STATIC_SMMR_DIR = PROJECT_ROOT_CLUSTER / "results" / "sensitivity" / "SMMR_phase"

# Dynamic (new) phase, percentile + slope, and merged SMMR record
DYNAMIC_THRESH_DIR = PROJECT_ROOT_CLUSTER / "results" / "dynamic_thresholds"

def load_static_phase_year(
    phase: str,
    year: int,
    window_days: int = 5,
) -> xr.DataArray:
    """
    Load old static phase (advance/retreat style) from
    seaice_phases_SMMR_YYYY_5day.nc etc.

    Parameters
    ----------
    phase : str
        e.g. "advance", "retreat" or whatever variable name is in the file.
    year : int
        e.g. 1979.
    window_days : int
        3, 5, or 7.

    Returns
    -------
    DataArray [y, x]
    """
    fname = STATIC_SMMR_DIR / f"seaice_phases_SMMR_{year}_{window_days}day.nc"
    ds = xr.open_dataset(fname)
    return ds[phase]


def load_dynamic_phase_climatology(
    phase: str,
    year_start: int,
    year_end: int,
    scheme: str = "quantile_slope",
    k: int = 5,
    p: float = 0.70,
    dC_min: float = 0.03,
) -> xr.DataArray:
    """
    Climatology of dynamic phase (FS/MS/ME) directly from dynamic-threshold files.

    Parameters
    ----------
    phase : {"FS", "MS", "ME"}
    year_start, year_end : int
        Range of years to include.
    scheme : str
        Dynamic scheme name, e.g. "quantile_slope".
    k : int
        Persistence window (must match what you used in run_dynamic_thresholds_staticSlope.py).
    p : float
        SIC percentile used in the dynamic threshold.
    dC_min : float
        Minimum daily SIC change used in the slope condition.

    Returns
    -------
    DataArray [y, x]
        Mean over years of the requested phase.
    """
    years = np.arange(year_start, year_end + 1)

    # example: .../dynamic_thresholds/quantile_slope_k5/FS/p0.7_dC_min0.03/
    tag = f"p{p}_dC_min{dC_min}"
    base_dir = (
        DYNAMIC_THRESH_DIR
        / f"{scheme}_k{k}"
        / phase
        / tag
    )

    fpaths = [base_dir / f"{phase}_{y}.nc" for y in years]

    ds = xr.open_mfdataset(fpaths, concat_dim="year", combine="nested")
    ds = ds.assign_coords(year=("year", years))
    da = ds[phase]
    return da.mean("year", skipna=True)



# ---------------------------------------------------------------------
# NAMING + PATH HELPERS
# ---------------------------------------------------------------------

def format_fig_name(num: int, short: str, ext: str = "png") -> str:
    """
    Standard figure name:
        FigNN_<short>.ext
    e.g. Fig03_FS_climatology_static_vs_dynamic.png
    """
    return f"Fig{num:02d}_{short}.{ext}"


def get_fig_path(
    project_root: Path | str,
    subfolder: str,
    fig_name: str,
    create: bool = True,
) -> Path:
    """
    Build a path under Results/Ch2_Figures/<subfolder>.

    Parameters
    ----------
    project_root : Path or str
        Path to repo root, e.g. Path("/user/geog/.../sea-ice-phase").
    subfolder : str
        e.g. "climatology", "sensitivity/window", "trends".
    fig_name : str
        Filename including extension.
    create : bool
        If True, mkdir the directory.

    Returns
    -------
    Path
    """
    root = Path(project_root)
    outdir = root / "results" / "Ch2_Figures" / subfolder
    if create:
        outdir.mkdir(parents=True, exist_ok=True)
    return outdir / fig_name


# ---------------------------------------------------------------------
# RCLOUD / GDRIVE UPLOAD
# ---------------------------------------------------------------------

def upload_with_rclone(
    local_path: Path | str,
    remote_root: str,
    subdir: Optional[str] = None,
    dry_run: bool = False,
    extra_flags: Optional[Iterable[str]] = None,
) -> None:
    """
    Mirror a local figure to Google Drive (or any rclone remote).

    Parameters
    ----------
    local_path : Path or str
        Local file to upload.
    remote_root : str
        e.g. "gdrive:sea-ice-phase/Results/Ch2_Figures".
    subdir : str, optional
        Extra subdirectory under remote_root
        (e.g. "climatology", "sensitivity/window").
    dry_run : bool
        If True, pass --dry-run to rclone.
    extra_flags : iterable of str, optional
        Additional rclone flags.

    Notes
    -----
    This is intentionally generic. Each figure script decides
    what remote_root/subdir to use.
    """
    lp = Path(local_path)
    if not lp.exists():
        print(f"[rclone] WARNING: local file does not exist: {lp}")
        return

    dst = remote_root.rstrip("/")
    if subdir:
        dst = f"{dst}/{subdir.strip('/')}"

    flags = list(extra_flags) if extra_flags else []
    cmd = ["rclone", "copy", str(lp), dst] + flags
    if dry_run:
        cmd.insert(1, "--dry-run")

    print("[rclone cmd]", " ".join(shlex.quote(c) for c in cmd))
    res = subprocess.run(cmd, text=True, capture_output=True)
    print("[rclone stdout]\n", res.stdout[-1000:])
    print("[rclone stderr]\n", res.stderr[-1000:])

    if res.returncode != 0:
        raise RuntimeError(f"rclone failed with code {res.returncode}")


def save_and_upload(
    fig: plt.Figure,
    out_path: Path | str,
    remote_root: Optional[str] = None,
    remote_subdir: Optional[str] = None,
    close: bool = True,
) -> None:
    """
    Save a figure to disk and optionally upload via rclone.

    remote_root example: "gdrive:sea-ice-phase/Results/Ch2_Figures"
    """
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    print(f"Saved: {out_path}")

    if remote_root is not None:
        upload_with_rclone(out_path, remote_root, subdir=remote_subdir)

    if close:
        plt.close(fig)


# ---------------------------------------------------------------------
# GENERIC DATA HELPERS
# ---------------------------------------------------------------------

def as_array(field: Any) -> np.ndarray:
    """Convert DataArray or array-like to plain np.ndarray."""
    if isinstance(field, xr.DataArray):
        return field.values
    return np.asarray(field)


def flatten_field(field: Any, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Flatten a 2D field, applying mask and removing NaNs.

    mask is True where valid (ocean), False elsewhere.
    """
    da = as_array(field)
    if mask is not None:
        valid = np.logical_and(~np.isnan(da), mask)
    else:
        valid = ~np.isnan(da)
    return da[valid]


# ---------------------------------------------------------------------
# POLAR MAP HELPERS
# ---------------------------------------------------------------------

def make_polar_axes(fig: plt.Figure, position: int, projection=None) -> plt.Axes:
    """
    Create a south polar stereographic axes with a circular boundary,
    land/ocean shading, and coastlines.

    position : subplot index, e.g. 1, 2, 3 in a 1x3 layout.
    """
    proj = projection or ccrs.SouthPolarStereo()
    ax = fig.add_subplot(1, 3, position, projection=proj)

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


def plot_phase_comparison_map(
    static_field,
    dynamic_field,
    lons=None,
    lats=None,
    label="Phase (day-of-year)",
    title_prefix="",
    diff_vlim=20,
    field_vmin=None,
    field_vmax=None,
):
    """
    Three-panel comparison: static, dynamic, dynamic-static.

    Handles two cases automatically:
      1) Geographical coords:  lon/lat + PlateCarree
      2) Native stereographic grid: x/y + SouthPolarStereo

    If lons/lats are not provided, it will try to infer them from the fields.
    """

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt

    proj = ccrs.SouthPolarStereo()

    # ---- Decide coordinate mode ----
    # Priority: explicit lons/lats, else from coords, else x/y native.
    if lons is not None and lats is not None:
        coord_mode = "geo"
    else:
        coords = set(static_field.coords)
        if {"lon", "lat"} <= coords:
            lons = static_field["lon"]
            lats = static_field["lat"]
            coord_mode = "geo"
        elif {"x", "y"} <= coords:
            # native stereographic grid
            x = static_field["x"]
            y = static_field["y"]
            coord_mode = "native"
        else:
            raise ValueError(
                "plot_phase_comparison_map needs lon/lat or x/y coords on the fields."
            )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12, 4.5),
        subplot_kw=dict(projection=proj),
        constrained_layout=True,
    )

    panels = [
        ("Static", static_field),
        ("Dynamic", dynamic_field),
        ("Difference", dynamic_field - static_field),
    ]

    for ax, (name, field) in zip(axes, panels):
        if coord_mode == "geo":
            im = ax.pcolormesh(
                lons,
                lats,
                field,
                transform=ccrs.PlateCarree(),
                cmap="RdBu_r" if name == "Difference" else "viridis",
                shading="auto",
                vmin=-diff_vlim if name == "Difference" else field_vmin,
                vmax=diff_vlim if name == "Difference" else field_vmax,
            )
        else:  # native stereographic x/y
            im = ax.pcolormesh(
                x,
                y,
                field,
                transform=proj,
                cmap="RdBu_r" if name == "Difference" else "viridis",
                shading="auto",
                vmin=-diff_vlim if name == "Difference" else field_vmin,
                vmax=diff_vlim if name == "Difference" else field_vmax,
            )

        # clean Antarctic map (no black ocean)
        ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="0.8", edgecolor="0.6", zorder=1)
        ax.coastlines(linewidth=0.4, zorder=2)

        ax.set_title(f"{title_prefix}{name}", fontsize=11, fontweight="bold")

        cbar = fig.colorbar(
            im,
            ax=ax,
            orientation="horizontal",
            pad=0.05,
            shrink=0.8,
        )
        if name == "Difference":
            cbar.set_label(f"{label} (dynamic − static)", fontsize=9)
        else:
            cbar.set_label(label, fontsize=9)
        cbar.ax.tick_params(labelsize=8)
        cbar.outline.set_visible(False)

    # Subfigure letters
    letters = ["(a)", "(b)", "(c)"]
    for letter, ax in zip(letters, axes):
        ax.text(
            0.02,
            0.98,
            letter,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=12,
            fontweight="bold",
        )

    return fig, axes


# ---------------------------------------------------------------------
# DISTRIBUTION HELPERS (CDF, sector CDF, time series)
# ---------------------------------------------------------------------

def plot_phase_cdf_comparison(
    static_field: Any,
    dynamic_field: Any,
    mask: Optional[np.ndarray] = None,
    title: str = "Phase date distribution",
    label_static: str = "Static",
    label_dynamic: str = "Dynamic",
    xlabel: str = "Day of year",
) -> tuple[plt.Figure, plt.Axes]:
    vals_s = flatten_field(static_field, mask=mask)
    vals_d = flatten_field(dynamic_field, mask=mask)

    fig, ax = plt.subplots(figsize=(4, 3), dpi=DPI)
    sns.ecdfplot(x=vals_s, ax=ax, label=label_static)
    sns.ecdfplot(x=vals_d, ax=ax, label=label_dynamic)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Cumulative probability")
    ax.set_title(title)
    ax.legend()

    return fig, ax


def plot_phase_cdf_by_sector(
    static_field: Any,
    dynamic_field: Any,
    sector_mask: np.ndarray,
    sector_ids: Iterable[int],
    sector_labels: Optional[Dict[int, str]] = None,
    phase_name: str = "FS",
    xlabel: str = "Day of year",
) -> tuple[plt.Figure, np.ndarray]:
    sector_ids = list(sector_ids)
    nsec = len(sector_ids)
    ncols = 3
    nrows = int(np.ceil(nsec / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(10, 3 * nrows),
        sharex=True,
        sharey=True,
        dpi=DPI,
    )
    axes = axes.ravel()

    if sector_labels is None:
        sector_labels = {i: f"Sector {i}" for i in sector_ids}

    for ax, sec in zip(axes, sector_ids):
        mask = (sector_mask == sec)

        vals_s = flatten_field(static_field, mask=mask)
        vals_d = flatten_field(dynamic_field, mask=mask)

        if vals_s.size == 0 or vals_d.size == 0:
            ax.set_visible(False)
            continue

        sns.ecdfplot(x=vals_s, ax=ax, label="Static")
        sns.ecdfplot(x=vals_d, ax=ax, label="Dynamic")
        ax.set_title(sector_labels.get(sec, f"Sector {sec}"))
        ax.grid(True, alpha=0.3)

    for ax in axes[:nsec]:
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Cumulative probability")

    for ax in axes[nsec:]:
        ax.set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2)

    fig.suptitle(f"{phase_name} date CDFs by sector (static vs dynamic)", y=0.98)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])

    return fig, axes
def plot_window_sensitivity_ecdf(
    adv_3,
    adv_5,
    adv_7,
    ret_3,
    ret_5,
    ret_7,
    mask: Optional[np.ndarray] = None,
    phase_label_advance: str = "Advance",
    phase_label_retreat: str = "Retreat",
    max_x: Optional[float] = None,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Plot ECDFs of |ΔDOY| sensitivity to running-window length (3 vs 5, 5 vs 7 days)
    for advance and retreat.

    Parameters
    ----------
    adv_3, adv_5, adv_7 : array-like or DataArray
        Climatological advance dates (DOY) for 3-, 5-, and 7-day windows.
    ret_3, ret_5, ret_7 : array-like or DataArray
        Climatological retreat dates (DOY) for 3-, 5-, and 7-day windows.
    mask : 2D bool array, optional
        True where ocean/valid. If None, all finite grid cells are used.
    phase_label_advance : str
        Label for the advance panel title.
    phase_label_retreat : str
        Label for the retreat panel title.
    max_x : float, optional
        Max x-limit for |ΔDOY| (days). If None, set to the 99th percentile
        across all four |Δ| fields.

    Returns
    -------
    fig : Figure
    axes : ndarray of Axes, shape (2,)
        axes[0] = advance ECDFs, axes[1] = retreat ECDFs.
    """
    # Convert to plain arrays
    adv_3 = as_array(adv_3)
    adv_5 = as_array(adv_5)
    adv_7 = as_array(adv_7)
    ret_3 = as_array(ret_3)
    ret_5 = as_array(ret_5)
    ret_7 = as_array(ret_7)

    # Absolute differences in days
    dadv_3v5 = np.abs(adv_3 - adv_5)
    dadv_5v7 = np.abs(adv_5 - adv_7)
    dret_3v5 = np.abs(ret_3 - ret_5)
    dret_5v7 = np.abs(ret_5 - ret_7)

    # Flatten with mask + NaN handling using existing helper
    adv_3v5_vals = flatten_field(dadv_3v5, mask=mask)
    adv_5v7_vals = flatten_field(dadv_5v7, mask=mask)
    ret_3v5_vals = flatten_field(dret_3v5, mask=mask)
    ret_5v7_vals = flatten_field(dret_5v7, mask=mask)

    # Optionally derive a sensible x-limit from the 99th percentile
    if max_x is None:
        all_vals = np.concatenate([
            adv_3v5_vals,
            adv_5v7_vals,
            ret_3v5_vals,
            ret_5v7_vals,
        ])
        # ignore NaNs
        all_vals = all_vals[np.isfinite(all_vals)]
        if all_vals.size > 0:
            max_x = np.nanpercentile(all_vals, 99.0)
        else:
            max_x = 10.0  # fallback

    fig, axes = plt.subplots(
        1,
        2,
        figsize=FIGSIZE_DOUBLE,
        dpi=DPI,
        sharey=True,
    )

    # --- Advance panel ---
    ax = axes[0]
    sns.ecdfplot(x=adv_3v5_vals, ax=ax, label="3 vs 5 days")
    sns.ecdfplot(x=adv_5v7_vals, ax=ax, label="5 vs 7 days")

    ax.set_xlabel(r"|Δ {} date| (days)".format(phase_label_advance.lower()))
    ax.set_ylabel("Cumulative probability")
    ax.set_xlim(0, max_x)
    #ax.set_title(phase_label_advance)

    # --- Retreat panel ---
    ax = axes[1]
    sns.ecdfplot(x=ret_3v5_vals, ax=ax, label="3 vs 5 days")
    sns.ecdfplot(x=ret_5v7_vals, ax=ax, label="5 vs 7 days")

    ax.set_xlabel(r"|Δ {} date| (days)".format(phase_label_retreat.lower()))
    ax.set_xlim(0, max_x)
    #ax.set_title(phase_label_retreat)

    # Shared legend (only once)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=2,
            frameon=False,
        )

    fig.tight_layout(rect=[0, 0.05, 1, 1])

    return fig, axes

def plot_window_sensitivity_ecdf(
    adv_3,
    adv_5,
    adv_7,
    ret_3,
    ret_5,
    ret_7,
    mask: Optional[np.ndarray] = None,
    phase_label_advance: str = "Advance",
    phase_label_retreat: str = "Retreat",
    max_x: Optional[float] = None,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Plot ECDFs of |ΔDOY| sensitivity to running-window length (3 vs 5, 5 vs 7 days)
    for advance and retreat.

    Parameters
    ----------
    adv_3, adv_5, adv_7 : array-like or DataArray
        Climatological advance dates (DOY) for 3-, 5-, and 7-day windows.
    ret_3, ret_5, ret_7 : array-like or DataArray
        Climatological retreat dates (DOY) for 3-, 5-, and 7-day windows.
    mask : 2D bool array, optional
        True where ocean/valid. If None, all finite grid cells are used.
    phase_label_advance : str
        Label for the advance panel title.
    phase_label_retreat : str
        Label for the retreat panel title.
    max_x : float, optional
        Max x-limit for |ΔDOY| (days). If None, set to the 99th percentile
        across all four |Δ| fields.

    Returns
    -------
    fig : Figure
    axes : ndarray of Axes, shape (2,)
        axes[0] = advance ECDFs, axes[1] = retreat ECDFs.
    """
    # Convert to plain arrays
    adv_3 = as_array(adv_3)
    adv_5 = as_array(adv_5)
    adv_7 = as_array(adv_7)
    ret_3 = as_array(ret_3)
    ret_5 = as_array(ret_5)
    ret_7 = as_array(ret_7)

    # Absolute differences in days
    dadv_3v5 = np.abs(adv_3 - adv_5)
    dadv_5v7 = np.abs(adv_5 - adv_7)
    dret_3v5 = np.abs(ret_3 - ret_5)
    dret_5v7 = np.abs(ret_5 - ret_7)

    # Flatten with mask + NaN handling using existing helper
    adv_3v5_vals = flatten_field(dadv_3v5, mask=mask)
    adv_5v7_vals = flatten_field(dadv_5v7, mask=mask)
    ret_3v5_vals = flatten_field(dret_3v5, mask=mask)
    ret_5v7_vals = flatten_field(dret_5v7, mask=mask)

    # Optionally derive a sensible x-limit from the 99th percentile
    if max_x is None:
        all_vals = np.concatenate([
            adv_3v5_vals,
            adv_5v7_vals,
            ret_3v5_vals,
            ret_5v7_vals,
        ])
        # ignore NaNs
        all_vals = all_vals[np.isfinite(all_vals)]
        if all_vals.size > 0:
            max_x = np.nanpercentile(all_vals, 99.0)
        else:
            max_x = 10.0  # fallback

    fig, axes = plt.subplots(
        1,
        2,
        figsize=FIGSIZE_DOUBLE,
        dpi=DPI,
        sharey=True,
    )

    # --- Advance panel ---
    ax = axes[0]
    sns.ecdfplot(x=adv_3v5_vals, ax=ax, label="3 vs 5 days")
    sns.ecdfplot(x=adv_5v7_vals, ax=ax, label="5 vs 7 days")

    ax.set_xlabel(r"|Δ {} date| (days)".format(phase_label_advance.lower()))
    ax.set_ylabel("Cumulative probability")
    ax.set_xlim(0, max_x)
    ax.set_title(phase_label_advance)

    # --- Retreat panel ---
    ax = axes[1]
    sns.ecdfplot(x=ret_3v5_vals, ax=ax, label="3 vs 5 days")
    sns.ecdfplot(x=ret_5v7_vals, ax=ax, label="5 vs 7 days")

    ax.set_xlabel(r"|Δ {} date| (days)".format(phase_label_retreat.lower()))
    ax.set_xlim(0, max_x)
    ax.set_title(phase_label_retreat)

    # Shared legend (only once)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=2,
            frameon=False,
        )

    fig.tight_layout(rect=[0, 0.05, 1, 1])

    return fig, axes


def plot_sector_time_series(
    anom_static: Any,
    anom_dynamic: Any,
    time: Any,
    sector_ids: Iterable[int],
    sector_labels: Optional[Dict[int, str]] = None,
    phase_name: str = "FS",
    ylabel: Optional[str] = None,
) -> tuple[plt.Figure, np.ndarray]:
    if isinstance(anom_static, xr.DataArray):
        static_vals = anom_static.transpose("time", "sector").values
        dynamic_vals = anom_dynamic.transpose("time", "sector").values
        sectors_dim = anom_static["sector"].values
    else:
        static_vals = np.asarray(anom_static)
        dynamic_vals = np.asarray(anom_dynamic)
        sectors_dim = np.asarray(list(sector_ids))

    sector_ids = list(sector_ids)
    nsec = len(sector_ids)
    ncols = 3
    nrows = int(np.ceil(nsec / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(10, 3 * nrows),
        sharex=True,
        sharey=True,
        dpi=DPI,
    )
    axes = axes.ravel()
    time = np.asarray(time)

    if sector_labels is None:
        sector_labels = {sec: f"Sector {sec}" for sec in sector_ids}
    if ylabel is None:
        ylabel = f"{phase_name} anomaly (days)"

    for ax, sec in zip(axes, sector_ids):
        try:
            idx = int(np.where(sectors_dim == sec)[0][0])
        except IndexError:
            ax.set_visible(False)
            continue

        sv = static_vals[:, idx]
        dv = dynamic_vals[:, idx]

        ax.plot(time, sv, label="Static", linewidth=1)
        ax.plot(time, dv, label="Dynamic", linewidth=1)
        ax.axhline(0.0, color="0.5", linewidth=0.5)
        ax.set_title(sector_labels.get(sec, f"Sector {sec}"))

    for ax in axes[:nsec]:
        ax.set_xlabel("Year")
        ax.set_ylabel(ylabel)

    for ax in axes[nsec:]:
        ax.set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2)

    fig.suptitle(f"{phase_name} anomalies by sector: static vs dynamic", y=0.98)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])

    return fig, axes


def get_sentinel_mask(project_root, phase: str, method: str = "static",
                      thr: int = 15, k: int = 5) -> "xr.DataArray":
    """
    Returns a boolean mask (y, x) — True where pixels should be masked out
    because they never undergo a genuine seasonal transition.

    After the sentinel-fix rerun of compute_phase_dates_v2.py, these pixels
    are NaN in the climatology file. We mask any pixel where the climatological
    mean is NaN (no valid phase date across the record).
    """
    import xarray as xr
    import numpy as np
    from pathlib import Path
    clim_file = (Path(project_root) / "data" / "anomalies" / "SMMR"
                 / f"{phase}_{method}_thr{thr}_k{k}_climatology.nc")
    ds = xr.open_dataset(clim_file, decode_times=False)
    varname = f"{phase}_{method}_thr{thr}_k{k}_clim"
    clim = ds[varname].load()
    ds.close()
    return np.isnan(clim)
