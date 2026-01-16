#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fig. 7 — Post-2017 step-change comparison (static vs dynamic) for FS/MS.

This script intentionally focuses on the *step change* (post − pre) and
removes the confusing linear-trend agreement panels.

Panels (2x3):
  (a) FS step-change sign agreement class (static vs dynamic)
  (b) FS sector mean step change (days)
  (c) FS method difference: (step change) dynamic − static (days)

  (d) MS step-change sign agreement class (static vs dynamic)
  (e) MS sector mean step change (days since Aug 15)
  (f) MS method difference: (step change) dynamic − static (days)

Pre/post periods:
  pre  = 1980–2017
  post = 2018–2023

MS handling:
  - Opens MS climatology/anomalies with decode_times=False to avoid CF-time decoding errors.
  - Auto-detects whether MS is already on "days since Aug 15" (0–~210) or calendar DOY (1–366).
  - If calendar DOY, wraps to a continuous axis and converts to days-since-Aug15 consistently.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ---------------------------------------------------------------------
# Shared utilities
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

PROJECT_ROOT = PROJECT_ROOT_CLUSTER
ANOM_DIR = PROJECT_ROOT / "results" / "anomalies"
SECTOR_FILE = PROJECT_ROOT / "data" / "canonical_sectors.nc"

REMOTE_ROOT = "gdrive:sea-ice-phase/Results/Ch2_Figures"
SUBFOLDER = "trends"  # keep your existing folder, unless you want a new one

# Step-change definition (post − pre)
PRE_START, PRE_END = 1980, 2017
POST_START, POST_END = 2018, 2023

# Canonical sectors (IDs are logic; labels are cosmetic)
sector_ids = [1, 2, 3, 4, 5]
sector_labels = {
    1: "A–B",   # Amundsen–Bellingshausen
    2: "WED",   # Weddell
    3: "KHV",   # King Haakon VII
    4: "EA",    # East Antarctica
    5: "RA",    # Ross–Amundsen (rename if you later change the mask)
}

# ---------------------------------------------------------------------
# Robust NetCDF variable opening
# ---------------------------------------------------------------------
def _open_var(path: Path, candidates: list[str], decode_times: bool = True) -> xr.DataArray:
    ds = xr.open_dataset(path, decode_times=decode_times)
    try:
        for v in candidates:
            if v in ds.data_vars:
                return ds[v].load()
        raise KeyError(f"No matching vars in {path.name}. Tried: {candidates}. Found: {list(ds.data_vars)}")
    finally:
        ds.close()


# ---------------------------------------------------------------------
# MS axis utilities
# ---------------------------------------------------------------------
def _looks_like_days_since_aug15(da: xr.DataArray) -> bool:
    """Heuristic: days-since-Aug15 should be ~0..210 (non-leap), with many NaNs."""
    v = da.values
    v = v[np.isfinite(v)]
    if v.size == 0:
        return False
    return (np.nanmin(v) >= -5) and (np.nanmax(v) <= 250)


def _wrap_ms_doy_to_aug15(doy: xr.DataArray, aug15_doy: int = 227) -> xr.DataArray:
    """
    Convert calendar DOY (1..366) to "days since Aug 15" (Aug 15 -> 0),
    wrapping across year boundary.
    """
    wrapped = xr.where(doy < aug15_doy, doy + 365, doy)
    return wrapped - aug15_doy


def ensure_ms_axis_days_since_aug15(ms_clim: xr.DataArray, ms_anom: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Ensure MS climatology and anomalies are in the same axis convention:
      - "days since Aug 15" (Aug15=0)

    If ms_clim already looks like 0..~210, assume it's already converted and leave as-is.
    If it looks like 1..366, convert ms_clim and *also* keep anomalies consistent:
        field(year) = clim + anom
      so if we transform clim by f(), we must also transform field then re-form anomalies,
      OR (since f is affine piecewise) we can safely transform clim+anom yearwise then re-anom.

    We do the safe thing: reconstruct field(year), transform, then recompute anomalies.
    """
    if _looks_like_days_since_aug15(ms_clim):
        # Already on days-since-Aug15 axis; assume anomalies match.
        print(f"[INFO] MS clim already looks like days since Aug 15 (min={float(ms_clim.min()):.1f}, max={float(ms_clim.max()):.1f})")
        return ms_clim, ms_anom

    # Otherwise treat as calendar DOY and convert consistently
    print("[INFO] MS clim does NOT look like days-since-Aug15; treating as calendar DOY and converting.")
    field = ms_clim + ms_anom  # [year,y,x] broadcast
    field_dsa = _wrap_ms_doy_to_aug15(field)
    clim_dsa = _wrap_ms_doy_to_aug15(ms_clim)
    anom_dsa = field_dsa - clim_dsa
    return clim_dsa, anom_dsa


# ---------------------------------------------------------------------
# Load fields + masks
# ---------------------------------------------------------------------
def load_fs_ms_clim_anom():
    fs_dyn_clim = _open_var(ANOM_DIR / "FS_dynamic_climatology.nc", ["FS_dynamic_clim"])
    fs_dyn_anom = _open_var(ANOM_DIR / "FS_dynamic_anomalies.nc", ["FS_dynamic_anom"])

    fs_sta_clim = _open_var(ANOM_DIR / "FS_static_climatology.nc", ["FS_static_clim"])
    fs_sta_anom = _open_var(ANOM_DIR / "FS_static_anomalies.nc", ["FS_static_anom"])

    # MS: always decode_times=False to avoid "days since Aug 15" CF decoding errors
    ms_dyn_clim = _open_var(
        ANOM_DIR / "MS_dynamic_climatology.nc",
        ["MS_dynamic_clim_dsa", "MS_dynamic_clim"],
        decode_times=False,
    )
    ms_dyn_anom = _open_var(
        ANOM_DIR / "MS_dynamic_anomalies.nc",
        ["MS_dynamic_anom_dsa", "MS_dynamic_anom"],
        decode_times=False,
    )

    ms_sta_clim = _open_var(
        ANOM_DIR / "MS_static_climatology.nc",
        ["MS_static_clim_dsa", "MS_static_clim"],
        decode_times=False,
    )
    ms_sta_anom = _open_var(
        ANOM_DIR / "MS_static_anomalies.nc",
        ["MS_static_anom_dsa", "MS_static_anom"],
        decode_times=False,
    )

    # Enforce consistent MS axis convention
    ms_dyn_clim, ms_dyn_anom = ensure_ms_axis_days_since_aug15(ms_dyn_clim, ms_dyn_anom)
    ms_sta_clim, ms_sta_anom = ensure_ms_axis_days_since_aug15(ms_sta_clim, ms_sta_anom)

    ds_mask = xr.open_dataset(SECTOR_FILE)
    try:
        valid_ocean = ds_mask["valid_ocean"].astype(bool).load()
        sector_mask = ds_mask["sector_id"].load()
    finally:
        ds_mask.close()

    return dict(
        FS_dynamic_clim=fs_dyn_clim,
        FS_dynamic_anom=fs_dyn_anom,
        FS_static_clim=fs_sta_clim,
        FS_static_anom=fs_sta_anom,
        MS_dynamic_clim=ms_dyn_clim,
        MS_dynamic_anom=ms_dyn_anom,
        MS_static_clim=ms_sta_clim,
        MS_static_anom=ms_sta_anom,
        valid_ocean=valid_ocean,
        sector_mask=sector_mask,
    )


# ---------------------------------------------------------------------
# Step-change math
# ---------------------------------------------------------------------
def compute_pre_post_diff(clim: xr.DataArray, anom: xr.DataArray) -> xr.DataArray:
    """
    Reconstruct mean fields for pre/post and return step change:
      diff = post_mean - pre_mean

    field(year) = clim + anom(year)
    """
    anom_pre = anom.sel(year=slice(PRE_START, PRE_END))
    anom_post = anom.sel(year=slice(POST_START, POST_END))

    pre_mean = clim + anom_pre.mean("year", skipna=True)
    post_mean = clim + anom_post.mean("year", skipna=True)
    return post_mean - pre_mean


def make_sign_agreement_class(diff_dyn: xr.DataArray, diff_sta: xr.DataArray, valid_ocean: xr.DataArray, thresh: float = 0.0) -> xr.DataArray:
    """
    Class codes (NaN over non-ocean):
      1 = both earlier        (dyn < -thresh AND sta < -thresh)
      2 = only dynamic earlier
      3 = only static earlier
      4 = both later          (dyn > +thresh AND sta > +thresh)

    Everything else (including near-zero mixed signs) -> NaN (not plotted)
    """
    dyn = diff_dyn.where(valid_ocean)
    sta = diff_sta.where(valid_ocean)

    cls = xr.full_like(dyn, np.nan, dtype=float)

    both_earlier = (dyn < -thresh) & (sta < -thresh)
    only_dyn = (dyn < -thresh) & (sta >= -thresh)
    only_sta = (sta < -thresh) & (dyn >= -thresh)
    both_later = (dyn > +thresh) & (sta > +thresh)

    cls = cls.where(~both_earlier, 1.0)
    cls = cls.where(~only_dyn, 2.0)
    cls = cls.where(~only_sta, 3.0)
    cls = cls.where(~both_later, 4.0)

    cls.name = "step_change_sign_agreement_class"
    return cls


def sector_mean_stepchange(diff_dyn: xr.DataArray, diff_sta: xr.DataArray, sector_mask: xr.DataArray, valid_ocean: xr.DataArray, phase: str) -> pd.DataFrame:
    recs = []
    for sec in sector_ids:
        mask = (sector_mask == sec) & valid_ocean
        d_dyn = float(np.nanmean(diff_dyn.where(mask).values))
        d_sta = float(np.nanmean(diff_sta.where(mask).values))
        recs.append(dict(phase=phase, sector_id=sec, sector_label=sector_labels[sec], method="Static", delta=d_sta))
        recs.append(dict(phase=phase, sector_id=sec, sector_label=sector_labels[sec], method="Dynamic", delta=d_dyn))
    return pd.DataFrame.from_records(recs)


# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------
def make_polar_ax(fig, gs, row, col):
    proj = ccrs.SouthPolarStereo()
    ax = fig.add_subplot(gs[row, col], projection=proj)
    ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="0.8", edgecolor="none", zorder=2)
    ax.set_facecolor("white")
    ax.gridlines(draw_labels=False, linewidth=0.3, color="0.7", alpha=0.4, linestyle="--")
    return ax


def plot_class_map(ax, da_class: xr.DataArray, title: str):
    """
    Plot sign-agreement classes (1..4). NaNs are transparent.
    """
    cmap = mcolors.ListedColormap(
        [
            "#2b8cbe",  # 1 both earlier
            "#41ab5d",  # 2 only dynamic earlier
            "#fdb462",  # 3 only static earlier
            "#d73027",  # 4 both later
        ]
    )
    bounds = [0.5, 1.5, 2.5, 3.5, 4.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    im = ax.pcolormesh(
        da_class["x"],
        da_class["y"],
        da_class,
        transform=ccrs.SouthPolarStereo(),
        cmap=cmap,
        norm=norm,
        shading="auto",
    )
    ax.set_title(title, fontsize=9, fontweight="bold")
    return im


def plot_continuous_map(ax, da: xr.DataArray, title: str, vlim: float = 20.0):
    """
    Diverging continuous map. Non-ocean should already be NaN.
    """
    im = ax.pcolormesh(
        da["x"],
        da["y"],
        da,
        transform=ccrs.SouthPolarStereo(),
        cmap="RdBu_r",
        vmin=-vlim,
        vmax=vlim,
        shading="auto",
    )
    ax.set_title(title, fontsize=9, fontweight="bold")
    return im


# ---------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------
def main():
    fields = load_fs_ms_clim_anom()
    valid_ocean = fields["valid_ocean"]
    sector_mask = fields["sector_mask"]

    # --- Step changes (post − pre) ---
    fs_diff_dyn = compute_pre_post_diff(fields["FS_dynamic_clim"], fields["FS_dynamic_anom"])
    fs_diff_sta = compute_pre_post_diff(fields["FS_static_clim"], fields["FS_static_anom"])

    ms_diff_dyn = compute_pre_post_diff(fields["MS_dynamic_clim"], fields["MS_dynamic_anom"])
    ms_diff_sta = compute_pre_post_diff(fields["MS_static_clim"], fields["MS_static_anom"])

    # --- Class maps ---
    fs_class = make_sign_agreement_class(fs_diff_dyn, fs_diff_sta, valid_ocean, thresh=0.0)
    ms_class = make_sign_agreement_class(ms_diff_dyn, ms_diff_sta, valid_ocean, thresh=0.0)

    # --- Sector means ---
    df_fs = sector_mean_stepchange(fs_diff_dyn, fs_diff_sta, sector_mask, valid_ocean, phase="FS")
    df_ms = sector_mean_stepchange(ms_diff_dyn, ms_diff_sta, sector_mask, valid_ocean, phase="MS")

    # --- Method-difference maps (dynamic − static) ---
    fs_method_diff = (fs_diff_dyn - fs_diff_sta).where(valid_ocean)
    ms_method_diff = (ms_diff_dyn - ms_diff_sta).where(valid_ocean)

    # --- Figure layout ---
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1.3, 1.0, 1.3])

    # Row 1: FS
    ax_a = make_polar_ax(fig, gs, 0, 0)
    im_class_fs = plot_class_map(ax_a, fs_class, f"(a) FS step-change sign agreement ({POST_START}–{POST_END} − {PRE_START}–{PRE_END})")

    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_title("(b) FS sector mean step change (days)", fontweight="bold", fontsize=9)

    ax_c = make_polar_ax(fig, gs, 0, 2)
    im_diff_fs = plot_continuous_map(ax_c, fs_method_diff, "(c) FS: (step change) dynamic − static (days)", vlim=20)

    # Row 2: MS
    ax_d = make_polar_ax(fig, gs, 1, 0)
    im_class_ms = plot_class_map(ax_d, ms_class, f"(d) MS step-change sign agreement ({POST_START}–{POST_END} − {PRE_START}–{PRE_END})")

    ax_e = fig.add_subplot(gs[1, 1])
    ax_e.set_title("(e) MS sector mean step change (days since Aug 15)", fontweight="bold", fontsize=9)

    ax_f = make_polar_ax(fig, gs, 1, 2)
    im_diff_ms = plot_continuous_map(ax_f, ms_method_diff, "(f) MS: (step change) dynamic − static (days)", vlim=20)

    # --- Bar plots (FS, MS) ---
    methods = ["Static", "Dynamic"]
    colors = {"Static": "#4575b4", "Dynamic": "#d73027"}
    x_pos = np.arange(len(sector_ids))
    width = 0.36

    # FS bars
    for i, m in enumerate(methods):
        dat = df_fs[df_fs["method"] == m].sort_values("sector_id")
        ax_b.bar(x_pos + (i - 0.5) * width, dat["delta"].values, width=width, color=colors[m], label=m)
    ax_b.axhline(0, color="0.4", linewidth=0.8)
    ax_b.set_xticks(x_pos)
    ax_b.set_xticklabels([sector_labels[s] for s in sector_ids])
    ax_b.set_ylabel("ΔFS (days)")
    ax_b.legend(frameon=True, fontsize=8)

    # MS bars
    for i, m in enumerate(methods):
        dat = df_ms[df_ms["method"] == m].sort_values("sector_id")
        ax_e.bar(x_pos + (i - 0.5) * width, dat["delta"].values, width=width, color=colors[m], label=m)
    ax_e.axhline(0, color="0.4", linewidth=0.8)
    ax_e.set_xticks(x_pos)
    ax_e.set_xticklabels([sector_labels[s] for s in sector_ids])
    ax_e.set_ylabel("ΔMS (days since Aug 15)")

    # --- Colorbars ---
    # Class colorbar (shared): show only classes 1..4
    cax1 = fig.add_axes([0.08, 0.06, 0.40, 0.02])
    cb1 = fig.colorbar(
        im_class_fs,
        cax=cax1,
        orientation="horizontal",
        ticks=[1, 2, 3, 4],
        boundaries=[0.5, 1.5, 2.5, 3.5, 4.5],
    )
    cb1.set_label("Step-change sign agreement class", fontsize=9)
    cb1.ax.set_xticklabels(["both earlier", "dyn earlier only", "static earlier only", "both later"], fontsize=7)
    cb1.outline.set_visible(False)

    # Continuous difference colorbar (shared)
    cax2 = fig.add_axes([0.56, 0.06, 0.36, 0.02])
    cb2 = fig.colorbar(im_diff_fs, cax=cax2, orientation="horizontal")
    cb2.set_label("(post−pre) dynamic − static (days)", fontsize=9)
    cb2.outline.set_visible(False)

    # Layout (avoid tight_layout Cartopy warning)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.95, bottom=0.11, wspace=0.25, hspace=0.20)

    fig_name = (
        f"Fig7_FS_MS_stepchange_static_vs_dynamic_"
        f"pre{PRE_START}-{PRE_END}_post{POST_START}-{POST_END}.png"
    )
    out_path = get_fig_path(PROJECT_ROOT, subfolder=SUBFOLDER, fig_name=fig_name)

    save_and_upload(fig, out_path, remote_root=REMOTE_ROOT, remote_subdir=SUBFOLDER)


if __name__ == "__main__":
    main()

