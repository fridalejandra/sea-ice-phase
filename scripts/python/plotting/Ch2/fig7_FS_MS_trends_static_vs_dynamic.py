#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fig. 7 — FS/MS timing changes: static vs dynamic.

Layout: 2 rows (FS, MS) x 3 cols:

Col 1: Step-change sign agreement (post–pre), categorical
    "earlier" = negative change; "later" = positive change
Col 2: Sector-mean step change (post–pre), static vs dynamic
Col 3: Trend agreement (linear): both methods have negative slope (earlier over time)

Periods:
  pre  = 1980–2017
  post = 2018–2023

IMPORTANT:
- FS is calendar day-of-year (days)
- MS is "days since Aug 15" (days) and should already be linear.
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
SUBFOLDER = "trends"

# Pre/post definition
PRE_START, PRE_END = 1980, 2017
POST_START, POST_END = 2018, 2023

# Canonical sectors: numeric IDs for logic, labels for plotting
sector_ids = [1, 2, 3, 4, 5]
sector_labels = {
    1: "A–B",   # Amundsen–Bellingshausen
    2: "WED",   # Weddell
    3: "KHV",   # King Haakon VII
    4: "EA",    # East Antarctica
    5: "RA",    # Ross–Amundsen (check your mask; rename label if you want)
}


# ---------------------------------------------------------------------
# Robust dataset loading helpers
# ---------------------------------------------------------------------
def _open_da(path: Path, candidates: list[str], decode_times: bool = True) -> xr.DataArray:
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")

    ds = xr.open_dataset(path, decode_times=decode_times)

    for name in candidates:
        if name in ds:
            da = ds[name].load()
            ds.close()
            return da

    vars_ = list(ds.data_vars)
    ds.close()
    raise KeyError(f"None of {candidates} found in {path}. Vars={vars_}")


def load_fs_ms_clim_anom() -> dict[str, xr.DataArray]:
    """
    Load FS/MS climatology + anomaly fields for static + dynamic,
    and the sector/ocean mask.
    """
    fs_dyn_clim = _open_da(
        ANOM_DIR / "FS_dynamic_climatology.nc",
        ["FS_dynamic_clim"],
        decode_times=True,
    )
    fs_dyn_anom = _open_da(
        ANOM_DIR / "FS_dynamic_anomalies.nc",
        ["FS_dynamic_anom"],
        decode_times=True,
    )

    # MS files may have a time-units attr that xarray tries to decode.
    ms_dyn_clim = _open_da(
        ANOM_DIR / "MS_dynamic_climatology.nc",
        ["MS_dynamic_clim_dsa", "MS_dynamic_clim"],
        decode_times=False,
    )
    ms_dyn_anom = _open_da(
        ANOM_DIR / "MS_dynamic_anomalies.nc",
        ["MS_dynamic_anom_dsa", "MS_dynamic_anom"],
        decode_times=False,
    )

    fs_sta_clim = _open_da(
        ANOM_DIR / "FS_static_climatology.nc",
        ["FS_static_clim"],
        decode_times=True,
    )
    fs_sta_anom = _open_da(
        ANOM_DIR / "FS_static_anomalies.nc",
        ["FS_static_anom"],
        decode_times=True,
    )

    ms_sta_clim = _open_da(
        ANOM_DIR / "MS_static_climatology.nc",
        ["MS_static_clim_dsa", "MS_static_clim"],
        decode_times=False,
    )
    ms_sta_anom = _open_da(
        ANOM_DIR / "MS_static_anomalies.nc",
        ["MS_static_anom_dsa", "MS_static_anom"],
        decode_times=False,
    )

    ds_mask = xr.open_dataset(SECTOR_FILE)
    valid_ocean = ds_mask["valid_ocean"].astype(bool)
    sector_mask = ds_mask["sector_id"]
    ds_mask.close()

    return {
        "FS_dynamic_clim": fs_dyn_clim,
        "FS_dynamic_anom": fs_dyn_anom,
        "MS_dynamic_clim": ms_dyn_clim,
        "MS_dynamic_anom": ms_dyn_anom,
        "FS_static_clim": fs_sta_clim,
        "FS_static_anom": fs_sta_anom,
        "MS_static_clim": ms_sta_clim,
        "MS_static_anom": ms_sta_anom,
        "valid_ocean": valid_ocean,
        "sector_mask": sector_mask,
    }


# ---------------------------------------------------------------------
# Core math: step change post−pre
# ---------------------------------------------------------------------
def compute_pre_post(clim: xr.DataArray, anom: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Reconstruct mean fields over pre and post periods:
      field(year) = clim + anom(year)
      pre_mean  = mean(field) over PRE_START..PRE_END
      post_mean = mean(field) over POST_START..POST_END
      diff      = post_mean − pre_mean
    """
    anom_pre = anom.sel(year=slice(PRE_START, PRE_END))
    anom_post = anom.sel(year=slice(POST_START, POST_END))

    pre_mean = clim + anom_pre.mean("year", skipna=True)
    post_mean = clim + anom_post.mean("year", skipna=True)
    diff = post_mean - pre_mean
    return pre_mean, post_mean, diff


def make_sign_class_map(diff_dyn: xr.DataArray, diff_sta: xr.DataArray, valid_ocean: xr.DataArray, thresh: float = 0.0) -> xr.DataArray:
    """
    Integer classification map:

      NaN = non-ocean
        1 = both earlier   (dyn < -thresh and sta < -thresh)
        2 = dyn earlier only
        3 = stat earlier only
        4 = both later     (dyn > +thresh and sta > +thresh)

    Note: "earlier" means negative step change (post−pre < 0).
    """
    dyn = diff_dyn.where(valid_ocean)
    sta = diff_sta.where(valid_ocean)

    cls = xr.full_like(dyn, fill_value=np.nan, dtype=float)

    both_earlier = (dyn < -thresh) & (sta < -thresh)
    dyn_only = (dyn < -thresh) & ~(sta < -thresh)
    sta_only = (sta < -thresh) & ~(dyn < -thresh)
    both_later = (dyn > +thresh) & (sta > +thresh)

    cls = xr.where(both_earlier, 1, cls)
    cls = xr.where(dyn_only, 2, cls)
    cls = xr.where(sta_only, 3, cls)
    cls = xr.where(both_later, 4, cls)

    cls.name = "step_sign_agreement"
    return cls


def sector_mean_deltas(diff_dyn_fs: xr.DataArray, diff_sta_fs: xr.DataArray,
                       diff_dyn_ms: xr.DataArray, diff_sta_ms: xr.DataArray,
                       sector_mask: xr.DataArray, valid_ocean: xr.DataArray) -> pd.DataFrame:
    """
    Sector-mean step change (post−pre) for dynamic and static, FS and MS.
    """
    records: list[dict] = []

    for phase, d_dyn, d_sta in [
        ("FS", diff_dyn_fs, diff_sta_fs),
        ("MS", diff_dyn_ms, diff_sta_ms),
    ]:
        for sec in sector_ids:
            mask = (sector_mask == sec) & valid_ocean
            dyn_mean = float(np.nanmean(d_dyn.where(mask).values))
            sta_mean = float(np.nanmean(d_sta.where(mask).values))
            records += [
                {"phase": phase, "sector_id": sec, "sector_label": sector_labels[sec], "method": "Static", "delta": sta_mean},
                {"phase": phase, "sector_id": sec, "sector_label": sector_labels[sec], "method": "Dynamic", "delta": dyn_mean},
            ]

    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------
# Core math: linear trends (agreement on negative slope)
# ---------------------------------------------------------------------
def _slope_from_series(y: np.ndarray, years: np.ndarray) -> float:
    m = np.isfinite(y)
    if m.sum() < 2:
        return np.nan
    return float(np.polyfit(years[m], y[m], 1)[0])


def compute_trend_slopes(anom_da: xr.DataArray) -> xr.DataArray:
    """
    slope (days/year) from anomalies time series at each gridcell.
    (Same slope as for the underlying field because clim is constant.)
    """
    years = anom_da["year"].values.astype(float)

    slopes = xr.apply_ufunc(
        _slope_from_series,
        anom_da,
        kwargs={"years": years},
        input_core_dims=[["year"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
    slopes.name = "slope_days_per_year"
    return slopes


def make_trend_agreement_mask(slope_dyn: xr.DataArray, slope_sta: xr.DataArray, valid_ocean: xr.DataArray) -> xr.DataArray:
    """
    Return a float mask suitable for plotting:
      1.0 where BOTH slopes < 0 (earlier over time)
      NaN elsewhere (including non-ocean)
    """
    agree = (slope_dyn < 0.0) & (slope_sta < 0.0) & valid_ocean
    out = xr.where(agree, 1.0, np.nan)
    out.name = "trend_agree_both_negative"
    return out


# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------
def make_polar_ax(fig, gs, row: int, col: int):
    proj = ccrs.SouthPolarStereo()
    ax = fig.add_subplot(gs[row, col], projection=proj)
    ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="0.8", edgecolor="none", zorder=2)
    ax.set_facecolor("white")
    ax.gridlines(draw_labels=False, linewidth=0.3, color="0.7", alpha=0.35, linestyle="--")
    return ax


def plot_sign_class_map(ax, da_class: xr.DataArray, title: str):
    """
    Categorical step-change sign agreement.
    """
    cmap = mcolors.ListedColormap(
        [
            "#2b8cbe",  # 1 both earlier
            "#41ab5d",  # 2 dyn earlier only
            "#fdb462",  # 3 stat earlier only
            "#d73027",  # 4 both later
        ]
    )
    bounds = [0.5, 1.5, 2.5, 3.5, 4.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Plot (NaNs are transparent)
    im = ax.pcolormesh(
        da_class["x"], da_class["y"], da_class,
        transform=ccrs.SouthPolarStereo(),
        cmap=cmap, norm=norm, shading="auto"
    )
    ax.set_title(title, fontsize=9, fontweight="bold")
    return im


def plot_trend_agreement(ax, agree_mask: xr.DataArray, title: str):
    """
    Plot NaN/1 mask: only show blue where both trends are earlier.
    """
    cmap = mcolors.ListedColormap(["#2b8cbe"])
    cmap.set_bad((1, 1, 1, 0))  # transparent for NaN

    im = ax.pcolormesh(
        agree_mask["x"], agree_mask["y"], agree_mask,
        transform=ccrs.SouthPolarStereo(),
        cmap=cmap, shading="auto", vmin=0, vmax=1
    )
    ax.set_title(title, fontsize=9, fontweight="bold")
    return im


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    fields = load_fs_ms_clim_anom()
    valid_ocean = fields["valid_ocean"]
    sector_mask = fields["sector_mask"]

    # Quick sanity prints for MS scale
    msd = fields["MS_dynamic_clim"].values
    mss = fields["MS_static_clim"].values
    print(f"[INFO] MS_dynamic_clim min/max: {np.nanmin(msd):.1f}/{np.nanmax(msd):.1f}")
    print(f"[INFO] MS_static_clim  min/max: {np.nanmin(mss):.1f}/{np.nanmax(mss):.1f}")

    # Step changes (post−pre)
    _, _, fs_dyn_diff = compute_pre_post(fields["FS_dynamic_clim"], fields["FS_dynamic_anom"])
    _, _, fs_sta_diff = compute_pre_post(fields["FS_static_clim"], fields["FS_static_anom"])
    _, _, ms_dyn_diff = compute_pre_post(fields["MS_dynamic_clim"], fields["MS_dynamic_anom"])
    _, _, ms_sta_diff = compute_pre_post(fields["MS_static_clim"], fields["MS_static_anom"])

    # Col 1: sign agreement classes (NaN outside ocean)
    fs_class = make_sign_class_map(fs_dyn_diff, fs_sta_diff, valid_ocean, thresh=0.0)
    ms_class = make_sign_class_map(ms_dyn_diff, ms_sta_diff, valid_ocean, thresh=0.0)

    # Col 2: sector mean bars
    df_sector = sector_mean_deltas(fs_dyn_diff, fs_sta_diff, ms_dyn_diff, ms_sta_diff, sector_mask, valid_ocean)

    # Col 3: trend agreement (both negative slopes)
    years = fields["FS_dynamic_anom"]["year"].values
    print(f"[INFO] Trend years span: {years.min()}–{years.max()}")

    fs_slope_dyn = compute_trend_slopes(fields["FS_dynamic_anom"])
    fs_slope_sta = compute_trend_slopes(fields["FS_static_anom"])
    ms_slope_dyn = compute_trend_slopes(fields["MS_dynamic_anom"])
    ms_slope_sta = compute_trend_slopes(fields["MS_static_anom"])

    fs_trend_agree = make_trend_agreement_mask(fs_slope_dyn, fs_slope_sta, valid_ocean)
    ms_trend_agree = make_trend_agreement_mask(ms_slope_dyn, ms_slope_sta, valid_ocean)

    # Fractions (relative to valid ocean gridcells)
    ocean_n = int(valid_ocean.values.sum())

    def frac_class(da, k):
        return float(np.isfinite(da.values).astype(int).sum() and np.nansum(da.values == k)) / ocean_n

    def frac_trend(da):
        return float(np.nansum(np.isfinite(da.values))) / ocean_n

    print("[INFO] Step-change BOTH EARLIER fraction (FS):", frac_class(fs_class, 1))
    print("[INFO] Step-change BOTH EARLIER fraction (MS):", frac_class(ms_class, 1))
    print("[INFO] Trend BOTH NEG SLOPE fraction (FS):", frac_trend(fs_trend_agree))
    print("[INFO] Trend BOTH NEG SLOPE fraction (MS):", frac_trend(ms_trend_agree))

    # ---------- Figure layout ----------
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1.25, 1.0, 1.25])

    fig.suptitle(f"Step change: {POST_START}–{POST_END} minus {PRE_START}–{PRE_END}", y=0.985, fontsize=11)

    # Panel titles (short, no bleeding)
    t_a = "(a) FS sign (post−pre)"
    t_b = "(b) FS sector mean Δ"
    t_c = "(c) FS trend agree (both earlier)"
    t_d = "(d) MS sign (post−pre)"
    t_e = "(e) MS sector mean Δ"
    t_f = "(f) MS trend agree (both earlier)"

    # ----- Row 1: FS -----
    ax_a = make_polar_ax(fig, gs, 0, 0)
    im_class = plot_sign_class_map(ax_a, fs_class, t_a)

    ax_b = fig.add_subplot(gs[0, 1])
    df_fs = df_sector[df_sector["phase"] == "FS"].copy()

    methods = ["Static", "Dynamic"]
    colors = {"Static": "#4575b4", "Dynamic": "#d73027"}
    x = np.arange(len(sector_ids))
    w = 0.36

    for i, method in enumerate(methods):
        dat = df_fs[df_fs["method"] == method].sort_values("sector_id")
        ax_b.bar(x + (i - 0.5) * w, dat["delta"].values, width=w, label=method, color=colors[method])

    ax_b.axhline(0, color="0.4", linewidth=0.8)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([sector_labels[s] for s in sector_ids])
    ax_b.set_ylabel("ΔFS (days)")
    ax_b.set_title(t_b, fontsize=9, fontweight="bold")
    ax_b.legend(frameon=True, fontsize=8)

    ax_c = make_polar_ax(fig, gs, 0, 2)
    im_trend = plot_trend_agreement(ax_c, fs_trend_agree, t_c)

    # ----- Row 2: MS -----
    ax_d = make_polar_ax(fig, gs, 1, 0)
    _ = plot_sign_class_map(ax_d, ms_class, t_d)

    ax_e = fig.add_subplot(gs[1, 1])
    df_ms = df_sector[df_sector["phase"] == "MS"].copy()

    for i, method in enumerate(methods):
        dat = df_ms[df_ms["method"] == method].sort_values("sector_id")
        ax_e.bar(x + (i - 0.5) * w, dat["delta"].values, width=w, color=colors[method])

    ax_e.axhline(0, color="0.4", linewidth=0.8)
    ax_e.set_xticks(x)
    ax_e.set_xticklabels([sector_labels[s] for s in sector_ids])
    ax_e.set_ylabel("ΔMS (days since Aug 15)")
    ax_e.set_title(t_e, fontsize=9, fontweight="bold")

    ax_f = make_polar_ax(fig, gs, 1, 2)
    _ = plot_trend_agreement(ax_f, ms_trend_agree, t_f)

    # ---------- Colorbars (compact, no "mask/land") ----------
    # Step-change sign agreement colorbar (class)
    cax1 = fig.add_axes([0.10, 0.06, 0.34, 0.015])
    cb1 = fig.colorbar(
        im_class,
        cax=cax1,
        orientation="horizontal",
        ticks=[1, 2, 3, 4],
    )
    cb1.set_label("Sign agreement class", fontsize=9)
    cb1.ax.set_xticklabels(["both −", "dyn − only", "stat − only", "both +"], fontsize=7)
    cb1.outline.set_visible(False)

    # Trend agreement legend (just a label; map is blue-only)
    # If you want a colorbar anyway, uncomment below.
    # cax2 = fig.add_axes([0.58, 0.06, 0.30, 0.015])
    # cb2 = fig.colorbar(im_trend, cax=cax2, orientation="horizontal")
    # cb2.set_ticks([])
    # cb2.set_label("Both methods: negative linear slope (earlier over time)", fontsize=9)
    # cb2.outline.set_visible(False)

    # Layout (avoid tight_layout with cartopy)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.93, bottom=0.11, wspace=0.25, hspace=0.26)

    # ---------- Save / upload ----------
    fig_name = (
        f"Fig7_FS_MS_stepchange_and_trend_static_vs_dynamic_"
        f"pre{PRE_START}-{PRE_END}_post{POST_START}-{POST_END}.png"
    )
    out_path = get_fig_path(PROJECT_ROOT, subfolder=SUBFOLDER, fig_name=fig_name)

    save_and_upload(fig, out_path, remote_root=REMOTE_ROOT, remote_subdir=SUBFOLDER)


if __name__ == "__main__":
    main()


