#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fig. 7 — FS/MS timing changes: static vs dynamic.

Layout: 2 rows (FS, MS) x 3 cols:

Col 1: Step-change sign agreement (post–pre), categorical
    "earlier" = negative change; "later" = positive change
Col 2: Sector-mean step change (post–pre), static vs dynamic
       **computed over ACTIVE pixels only**
Col 3: Trend agreement (linear): both methods have negative slope (earlier over time)
       **restricted to ACTIVE pixels only**

Periods:
  pre  = 1979–2015
  post = 2016–2024

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

from utils.plot_utils import (  # noqa: E402
    set_mpl_defaults,
    get_fig_path,
    save_and_upload,
    PROJECT_ROOT_CLUSTER,
)

set_mpl_defaults()

PROJECT_ROOT = PROJECT_ROOT_CLUSTER
ANOM_DIR = PROJECT_ROOT / "data" / "anomalies" / "SMMR"
SECTOR_FILE = PROJECT_ROOT / "data" / "canonical_sectors.nc"

REMOTE_ROOT = "gdrive:sea-ice-phase/results/Ch2_Figures"
SUBFOLDER = ""

# Pre/post definition
PRE_START, PRE_END = 1979, 2015
POST_START, POST_END = 2016, 2024

# Canonical sectors
sector_ids = [1, 2, 3, 4, 5]
sector_labels = {
    1: "A–B",   # Amundsen–Bellingshausen
    2: "WED",   # Weddell
    3: "KHV",   # King Haakon VII
    4: "EA",    # East Antarctica
    5: "RA",    # Ross–Amundsen (rename if mask differs)
}

# Active pixel criterion (applies to Col 2 + Col 3)
MIN_FRAC_ACTIVE = 0.80  # require valid timing in >=80% of years for BOTH methods


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
    fs_dyn_clim = _open_da(ANOM_DIR / "FS_dynamic_k5_q70_climatology.nc", ["FS_dynamic_k5_q70_clim"], decode_times=False)
    fs_dyn_anom = _open_da(ANOM_DIR / "FS_dynamic_k5_q70_anomalies.nc", ["FS_dynamic_k5_q70_anom"], decode_times=False)

    # MS NetCDFs may have a time-units attr xarray tries to decode -> decode_times=False
    ms_dyn_clim = _open_da(ANOM_DIR / "MS_dynamic_k5_q70_climatology.nc", ["MS_dynamic_k5_q70_clim_dsa", "MS_dynamic_k5_q70_clim"], decode_times=False)
    ms_dyn_anom = _open_da(ANOM_DIR / "MS_dynamic_k5_q70_anomalies.nc", ["MS_dynamic_k5_q70_anom_dsa", "MS_dynamic_k5_q70_anom"], decode_times=False)

    fs_sta_clim = _open_da(ANOM_DIR / "FS_static_thr15_k5_climatology.nc", ["FS_static_thr15_k5_clim"], decode_times=False)
    fs_sta_anom = _open_da(ANOM_DIR / "FS_static_thr15_k5_anomalies.nc", ["FS_static_thr15_k5_anom"], decode_times=False)

    ms_sta_clim = _open_da(ANOM_DIR / "MS_static_thr15_k5_climatology.nc", ["MS_static_thr15_k5_clim_dsa", "MS_static_thr15_k5_clim"], decode_times=False)
    ms_sta_anom = _open_da(ANOM_DIR / "MS_static_thr15_k5_anomalies.nc", ["MS_static_thr15_k5_anom_dsa", "MS_static_thr15_k5_anom"], decode_times=False)

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
# Active pixel masks (per phase; BOTH methods must be active)
# ---------------------------------------------------------------------
def make_activity_mask(
    anom_dyn: xr.DataArray,
    anom_sta: xr.DataArray,
    valid_ocean: xr.DataArray,
    frac_required: float = 0.80,
) -> xr.DataArray:
    """
    Active pixel definition:
      active = finite values in BOTH methods for >= frac_required of years.

    Inputs are anomalies [year,y,x].
    Returns boolean [y,x].
    """
    n_years = float(anom_dyn.sizes["year"])
    dyn_frac = anom_dyn.notnull().sum("year") / n_years
    sta_frac = anom_sta.notnull().sum("year") / n_years
    active = (dyn_frac >= frac_required) & (sta_frac >= frac_required) & valid_ocean
    active.name = "active_mask"
    return active


# ---------------------------------------------------------------------
# Step change post−pre
# ---------------------------------------------------------------------
def compute_pre_post(
    clim: xr.DataArray,
    anom: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
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


def make_sign_class_map(
    diff_dyn: xr.DataArray,
    diff_sta: xr.DataArray,
    valid_ocean: xr.DataArray,
    thresh: float = 0.0,
) -> xr.DataArray:
    """
    Categorical agreement map for step change (post−pre):

      NaN = non-ocean
        1 = both earlier   (dyn < -thresh AND sta < -thresh)
        2 = dyn earlier only
        3 = stat earlier only
        4 = both later     (dyn > +thresh AND sta > +thresh)
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


def sector_mean_deltas(
    diff_dyn_fs: xr.DataArray,
    diff_sta_fs: xr.DataArray,
    diff_dyn_ms: xr.DataArray,
    diff_sta_ms: xr.DataArray,
    sector_mask: xr.DataArray,
    valid_ocean: xr.DataArray,
    fs_active: xr.DataArray,
    ms_active: xr.DataArray,
) -> pd.DataFrame:
    """
    Sector-mean step change (post−pre) for dynamic and static, restricted to ACTIVE pixels.
    """
    records: list[dict] = []

    phase_specs = [
        ("FS", diff_dyn_fs, diff_sta_fs, fs_active),
        ("MS", diff_dyn_ms, diff_sta_ms, ms_active),
    ]

    for phase, d_dyn, d_sta, active in phase_specs:
        for sec in sector_ids:
            mask = (sector_mask == sec) & valid_ocean & active

            dyn_mean = float(np.nanmean(d_dyn.where(mask).values))
            sta_mean = float(np.nanmean(d_sta.where(mask).values))

            records.append({"phase": phase, "sector_id": sec, "sector_label": sector_labels[sec], "method": "Static", "delta": sta_mean})
            records.append({"phase": phase, "sector_id": sec, "sector_label": sector_labels[sec], "method": "Dynamic", "delta": dyn_mean})

    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------
# Linear trends (agreement on negative slope)
# ---------------------------------------------------------------------
def _slope_from_series(y: np.ndarray, years: np.ndarray) -> float:
    m = np.isfinite(y)
    if m.sum() < 2:
        return np.nan
    return float(np.polyfit(years[m], y[m], 1)[0])


def compute_trend_slopes(anom_da: xr.DataArray) -> xr.DataArray:
    """
    slope (days/year) from anomalies time series at each gridcell.
    (Same slope as the underlying field because clim is constant.)
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


def make_trend_agreement_mask(
    slope_dyn: xr.DataArray,
    slope_sta: xr.DataArray,
    valid_ocean: xr.DataArray,
    active_mask: xr.DataArray,
) -> xr.DataArray:
    """
    Return a float mask suitable for plotting:
      1.0 where BOTH slopes < 0 (earlier over time) AND pixel is ACTIVE
      NaN elsewhere (including non-ocean and inactive)
    """
    mask = valid_ocean & active_mask
    agree = (slope_dyn < 0.0) & (slope_sta < 0.0) & mask
    out = xr.where(agree, 1.0, np.nan)
    out.name = "trend_agree_both_negative_active"
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

    im = ax.pcolormesh(
        da_class["x"], da_class["y"], da_class,
        transform=ccrs.SouthPolarStereo(),
        cmap=cmap, norm=norm, shading="auto"
    )
    ax.set_title(title, fontsize=9, fontweight="bold")
    return im


def plot_trend_agreement(ax, agree_mask: xr.DataArray, title: str):
    cmap = mcolors.ListedColormap(["#2b8cbe"])
    cmap.set_bad((1, 1, 1, 0))  # transparent NaN

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

    # ACTIVE masks (used for Col 2 + Col 3)
    fs_active = make_activity_mask(fields["FS_dynamic_anom"], fields["FS_static_anom"], valid_ocean, frac_required=MIN_FRAC_ACTIVE)
    ms_active = make_activity_mask(fields["MS_dynamic_anom"], fields["MS_static_anom"], valid_ocean, frac_required=MIN_FRAC_ACTIVE)
    print(f"[INFO] Active mask @ {MIN_FRAC_ACTIVE:.2f}: FS={int(fs_active.values.sum())}, MS={int(ms_active.values.sum())}")

    # Col 1: sign agreement classes (NaN outside ocean)
    fs_class = make_sign_class_map(fs_dyn_diff.where(fs_active), fs_sta_diff.where(fs_active), valid_ocean, thresh=7.0)
    ms_class = make_sign_class_map(ms_dyn_diff, ms_sta_diff, valid_ocean, thresh=7.0)

    # Col 2: sector mean bars (ACTIVE-only)
    df_sector = sector_mean_deltas(
        fs_dyn_diff, fs_sta_diff,
        ms_dyn_diff, ms_sta_diff,
        sector_mask, valid_ocean,
        fs_active, ms_active
    )

    # Col 3: trend agreement (ACTIVE-only)
    years = fields["FS_dynamic_anom"]["year"].values
    print(f"[INFO] Trend years span: {years.min()}–{years.max()}")

    fs_slope_dyn = compute_trend_slopes(fields["FS_dynamic_anom"])
    fs_slope_sta = compute_trend_slopes(fields["FS_static_anom"])
    ms_slope_dyn = compute_trend_slopes(fields["MS_dynamic_anom"])
    ms_slope_sta = compute_trend_slopes(fields["MS_static_anom"])

    fs_trend_agree = make_trend_agreement_mask(fs_slope_dyn, fs_slope_sta, valid_ocean, fs_active)
    ms_trend_agree = make_trend_agreement_mask(ms_slope_dyn, ms_slope_sta, valid_ocean, ms_active)

    # ---------------- Fractions (make denominators explicit) ----------------
    def frac_step_both_earlier_active(da_class: xr.DataArray, active_mask: xr.DataArray) -> float:
        denom = int((valid_ocean & active_mask).values.sum())
        if denom == 0:
            return np.nan
        # class==1 among ACTIVE pixels
        num = int(np.nansum((da_class.where(active_mask)).values == 1))
        return num / float(denom)

    def frac_trend_agree_active(agree_mask: xr.DataArray, active_mask: xr.DataArray) -> float:
        denom = int((valid_ocean & active_mask).values.sum())
        if denom == 0:
            return np.nan
        num = int(np.nansum(agree_mask.values == 1.0))
        return num / float(denom)

    print(f"[INFO] Step-change BOTH EARLIER fraction (FS) [denom=ACTIVE @ {MIN_FRAC_ACTIVE:.2f}]:",
          frac_step_both_earlier_active(fs_class, fs_active))
    print(f"[INFO] Step-change BOTH EARLIER fraction (MS) [denom=ACTIVE @ {MIN_FRAC_ACTIVE:.2f}]:",
          frac_step_both_earlier_active(ms_class, ms_active))
    print(f"[INFO] Trend BOTH NEG SLOPE fraction (FS) [denom=ACTIVE @ {MIN_FRAC_ACTIVE:.2f}]:",
          frac_trend_agree_active(fs_trend_agree, fs_active))
    print(f"[INFO] Trend BOTH NEG SLOPE fraction (MS) [denom=ACTIVE @ {MIN_FRAC_ACTIVE:.2f}]:",
          frac_trend_agree_active(ms_trend_agree, ms_active))

    def frac_trend_both_positive_active(
        slope_dyn: xr.DataArray, slope_sta: xr.DataArray, active_mask: xr.DataArray
    ) -> float:
        """Fraction of ACTIVE pixels where BOTH methods have positive slope (later over time)."""
        denom = int((valid_ocean & active_mask).values.sum())
        if denom == 0:
            return np.nan
        both_pos = (slope_dyn > 0.0) & (slope_sta > 0.0) & valid_ocean & active_mask
        num = int(both_pos.values.sum())
        return num / float(denom)
    print(f"[INFO] Trend BOTH POS SLOPE fraction (FS) [denom=ACTIVE @ {MIN_FRAC_ACTIVE:.2f}]:",
          frac_trend_both_positive_active(fs_slope_dyn, fs_slope_sta, fs_active))
    print(f"[INFO] Trend BOTH POS SLOPE fraction (MS) [denom=ACTIVE @ {MIN_FRAC_ACTIVE:.2f}]:",
          frac_trend_both_positive_active(ms_slope_dyn, ms_slope_sta, ms_active))

    def frac_step_both_later_active(da_class: xr.DataArray, active_mask: xr.DataArray) -> float:
        denom = int((valid_ocean & active_mask).values.sum())
        if denom == 0:
            return np.nan
        num = int(np.nansum((da_class.where(active_mask)).values == 4))
        return num / float(denom)
    print(f"[INFO] Step-change BOTH LATER fraction (FS) [denom=ACTIVE @ {MIN_FRAC_ACTIVE:.2f}]:",
          frac_step_both_later_active(fs_class, fs_active))
    print(f"[INFO] Step-change BOTH LATER fraction (MS) [denom=ACTIVE @ {MIN_FRAC_ACTIVE:.2f}]:",
          frac_step_both_later_active(ms_class, ms_active))

    # ---------------- Figure layout ----------------
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1.25, 1.0, 1.25])

    fig.suptitle("Post-2016 timing shift (2016–2024 minus 1979–2015)", y=0.985, fontsize=11)

    # Short panel titles (no bleeding)
    t_a = "(a) FS sign (post−pre)"
    t_b = "(b) FS sector Δ"
    t_c = "(c) FS trend agree"
    t_d = "(d) MS sign (post−pre)"
    t_e = "(e) MS sector Δ"
    t_f = "(f) MS trend agree"

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
    _ = plot_trend_agreement(ax_c, fs_trend_agree, t_c)

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

    # ---------- Colorbar (compact; no land/mask label) ----------
    cax1 = fig.add_axes([0.075, 0.055, 0.26, 0.012])  # [left, bottom, width, height]
    cb1 = fig.colorbar(im_class, cax=cax1, orientation="horizontal", ticks=[1, 2, 3, 4])
    cb1.set_label("Sign agreement class", fontsize=9)
    cb1.ax.set_xticklabels(["both −", "dyn − only", "stat − only", "both +"], fontsize=7)
    cb1.outline.set_visible(False)

    # Layout (avoid tight_layout with cartopy)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.93, bottom=0.11, wspace=0.25, hspace=0.26)

    # ---------- Save / upload ----------
    fig_name = (
        f"Fig08_FS_MS_stepchange_and_trend_static_vs_dynamic_"
        f"pre{PRE_START}-{PRE_END}_post{POST_START}-{POST_END}_active{int(MIN_FRAC_ACTIVE*100)}.png"
    )
    out_path = get_fig_path(PROJECT_ROOT, subfolder=SUBFOLDER, fig_name=fig_name)

    save_and_upload(fig, out_path, remote_root=REMOTE_ROOT, remote_subdir="")


if __name__ == "__main__":
    main()
