#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig_FS_MS_anomaly_timeseries_static_dynamic_masks.py

Handcock-style anomaly time series for FS/MS, comparing:
  - Static vs dynamic methods
  - Seasonal ice zone vs all-ice zone (for diagnostics)

Inputs (already produced):
    results/anomalies/FS_dynamic_anomalies.nc
    results/anomalies/MS_dynamic_anomalies.nc
    results/anomalies/FS_static_anomalies.nc
    results/anomalies/MS_static_anomalies.nc

Masks:
    - All-ice: valid_ocean from data/canonical_sectors.nc
    - Seasonal ice zone: derived from Bootstrap SIC
        data/bootstrap_smmr/merged_bootstrap_SH_1979_20251001.nc

Main output:
    results/Ch2_Figures/anomalies/Fig_FS_MS_static_dynamic_anomaly_timeseries.png
      (3-panels, seasonal ice zone only, dynamic vs static)

Optional diagnostic output:
    results/Ch2_Figures/anomalies/Fig_FS_MS_seasonal_vs_all_anomaly_timeseries.png
"""

import sys
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import statsmodels.api as sm

# ---------------------------------------------------------------------
# Import shared utils
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
BOOTSTRAP_FILE = PROJECT_ROOT / "data" / "bootstrap_smmr" / "merged_bootstrap_SH_1979_20251001.nc"

REMOTE_ROOT = "gdrive:sea-ice-phase/Results/Ch2_Figures"
SUBFOLDER = "anomalies"

YEARS_HI = [2016, 2022, 2023]
YEAR_SPLIT = 2016  # pre- vs post-2016

# ---------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------

def _load_anom(fname: str, preferred_names) -> xr.DataArray:
    """Robust loader: try preferred_names, else first data_var."""
    ds = xr.open_dataset(ANOM_DIR / fname)
    for name in preferred_names:
        if name in ds.data_vars:
            return ds[name]
    # fallback: first data variable
    vname = list(ds.data_vars)[0]
    return ds[vname]


def load_all_anoms():
    # FS
    fs_dyn = _load_anom("FS_dynamic_anomalies.nc",
                        ["FS_dynamic_anom", "FS_anom", "FS"])
    fs_stat = _load_anom("FS_static_anomalies.nc",
                         ["FS_static_anom", "FS_anom", "FS"])

    # MS
    ms_dyn = _load_anom("MS_dynamic_anomalies.nc",
                        ["MS_dynamic_anom", "MS_anom", "MS"])
    ms_stat = _load_anom("MS_static_anomalies.nc",
                         ["MS_static_anom", "MS_anom", "MS"])

    return fs_dyn, fs_stat, ms_dyn, ms_stat


# ---------------------------------------------------------------------
# Masks: all-ice vs seasonal ice zone
# ---------------------------------------------------------------------

def build_masks():
    """Return (mask_all, mask_seasonal, area) as np.ndarray.

    mask_all: valid ocean grid
    mask_seasonal: cells that are seasonally ice-covered based on Bootstrap SIC
                   (here: fraction of days with SIC>0.15 between 0.05 and 0.95)

    This is intentionally simple and tunable.
    """
    sec = xr.open_dataset(SECTOR_FILE)
    valid = sec["valid_ocean"].astype(bool)
    area = sec["area_m2"]

    mask_all = valid.values

    # --- seasonal mask from Bootstrap ---
    boot = xr.open_dataset(BOOTSTRAP_FILE)
    sic = boot["N07_ICECON"]

    # unpack scale/offset if present (Bootstrap style)
    scale = sic.attrs.get("scale_factor", 1.0)
    offset = sic.attrs.get("add_offset", 0.0)
    sic = sic.astype("float32") * scale + offset

    # Convert to fraction 0–1 if units are 0–100
    if sic.max() > 1.5:
        sic = sic / 100.0

    # restrict to 1980–2016 for defining the climatological seasonal zone
    # time is "days since 1970-01-01"
    t = boot["time"]
    years = xr.cftime_range("1970-01-01", periods=t.size, freq="D").year
    boot = boot.assign_coords(year=("time", years))
    sic = sic.assign_coords(year=("time", years))
    sic_ref = sic.where((sic["year"] >= 1980) & (sic["year"] <= 2016), drop=True)

    # fraction of days with SIC > 0.15
    frac_ice = (sic_ref > 0.15).mean("time")

    # define seasonal: neither almost-always ice-free nor almost-always ice-covered
    seasonal = (frac_ice > 0.05) & (frac_ice < 0.95) & valid

    mask_seasonal = seasonal.values
    area = area.where(valid).values

    return mask_all, mask_seasonal, area


# ---------------------------------------------------------------------
# Area-weighted series + LOESS + trends
# ---------------------------------------------------------------------

def area_weighted_series(da: xr.DataArray, mask: np.ndarray, area: np.ndarray):
    """Return years, weighted spatial mean (np.ndarray)."""
    # da dims: (year, y, x)
    years = da["year"].values

    w = np.where(mask, area, np.nan)
    w = w / np.nansum(w)

    vals = []
    for yy in years:
        field = da.sel(year=yy).values
        vals.append(np.nansum(field * w))
    return years, np.array(vals)


def loess_smooth(x, y, frac=0.25):
    lo = sm.nonparametric.lowess(y, x, frac=frac, return_sorted=True)
    return lo[:, 0], lo[:, 1]


def linear_trend_with_ci(years, series, start, end):
    """Simple OLS trend (days/decade) with 95% CI."""
    mask = (years >= start) & (years <= end)
    x = years[mask].astype(float)
    y = series[mask].astype(float)

    if x.size < 3 or np.all(~np.isfinite(y)):
        return np.nan, np.nan, np.nan

    x_center = x - x.mean()
    # slope, intercept on centered x
    b1, b0 = np.polyfit(x_center, y, 1)

    y_hat = b0 + b1 * x_center
    resid = y - y_hat
    dof = x.size - 2
    if dof <= 0:
        return np.nan, np.nan, np.nan

    sigma2 = np.nansum(resid ** 2) / dof
    Sxx = np.nansum(x_center ** 2)
    se_b1 = np.sqrt(sigma2 / Sxx)

    # 95% CI using normal approx (fine for n>20)
    z = 1.96
    slope_dec = b1 * 10.0               # days/decade
    ci_low = (b1 - z * se_b1) * 10.0
    ci_high = (b1 + z * se_b1) * 10.0

    return slope_dec, ci_low, ci_high


# ---------------------------------------------------------------------
# Main plotting / analysis
# ---------------------------------------------------------------------

def main():
    print("Loading anomalies...")
    fs_dyn, fs_stat, ms_dyn, ms_stat = load_all_anoms()
    mask_all, mask_seasonal, area = build_masks()

    # area-weighted series for each combo
    series = {}

    for label, da in [
        ("FS_dynamic", fs_dyn),
        ("FS_static", fs_stat),
        ("MS_dynamic", ms_dyn),
        ("MS_static", ms_stat),
    ]:
        yrs, s_all = area_weighted_series(da, mask_all, area)
        _, s_seas = area_weighted_series(da, mask_seasonal, area)
        series[(label, "all")] = (yrs, s_all)
        series[(label, "seasonal")] = (yrs, s_seas)

    # sanity: all years identical
    years = series[("FS_dynamic", "seasonal")][0]

    # build duration = MS - FS for each method/mask
    for method in ["dynamic", "static"]:
        for mtype in ["all", "seasonal"]:
            yrs_fs, fs = series[(f"FS_{method}", mtype)]
            yrs_ms, ms = series[(f"MS_{method}", mtype)]
            assert np.all(yrs_fs == yrs_ms)
            dur = ms - fs
            series[(f"DUR_{method}", mtype)] = (yrs_fs, dur)

    # -----------------------------------------------------------------
    # A: main figure – seasonal mask only, dynamic vs static
    # -----------------------------------------------------------------
    print("Building main seasonal-ice figure (dynamic vs static)...")

    fig, axes = plt.subplots(3, 1, figsize=(7.5, 6.0), sharex=True)
    fig.patch.set_facecolor("white")

    panels = [
        ("(a) Season duration anomaly (days)", "DUR"),
        ("(b) Retreat anomaly (days)", "MS"),
        ("(c) Advance anomaly (days)", "FS"),
    ]

    for ax, (title, phase_tag) in zip(axes, panels):
        # dynamic seasonal
        yrs, dyn = series[(f"{phase_tag}_dynamic", "seasonal")]
        _, dyn_smooth = loess_smooth(yrs, dyn)

        # static seasonal
        _, stat = series[(f"{phase_tag}_static", "seasonal")]
        _, stat_smooth = loess_smooth(yrs, stat)

        # sigma band from dynamic (seasonal)
        sigma = np.nanstd(dyn)

        # grey band
        ax.fill_between(
            yrs,
            dyn_smooth - sigma,
            dyn_smooth + sigma,
            color="0.9",
            zorder=0,
        )

        # raw yearly dynamic in light grey
        ax.plot(yrs, dyn, color="0.7", linewidth=0.9, zorder=1)

        # smooth lines
        ax.plot(yrs, dyn_smooth, color="black", linewidth=1.8, label="Dynamic")
        ax.plot(yrs, stat_smooth, color="#1f77b4", linewidth=1.8, label="Static")

        # highlight years (dynamic)
        for yy in YEARS_HI:
            if yy in yrs:
                val = dyn[np.where(yrs == yy)][0]
                ax.scatter(yy, val, color="red", s=30, zorder=5)

        # vertical line at 2016
        if YEAR_SPLIT in yrs:
            ax.axvline(YEAR_SPLIT, color="red", linestyle="--",
                       linewidth=1.0, alpha=0.7)

        ax.set_ylabel("Days", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(alpha=0.25, linewidth=0.5)

    axes[-1].set_xlabel("Year", fontsize=9)

    # single legend in top panel
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc="upper left", frameon=False)

    fig.tight_layout()

    outpath = get_fig_path(
        project_root=PROJECT_ROOT,
        subfolder=SUBFOLDER,
        fig_name="Fig_FS_MS_static_dynamic_anomaly_timeseries.png",
    )
    save_and_upload(
        fig,
        outpath,
        remote_root=REMOTE_ROOT,
        remote_subdir=SUBFOLDER,
    )

    # -----------------------------------------------------------------
    # B/C: print trends (pre-2016 vs post-2016) for seasonal mask
    # -----------------------------------------------------------------
    print("\n=== Linear trends (seasonal ice zone, days/decade) ===")
    for phase_tag, label in [("DUR", "Season duration"),
                             ("MS", "Retreat"),
                             ("FS", "Advance")]:
        yrs, dyn = series[(f"{phase_tag}_dynamic", "seasonal")]
        _, stat = series[(f"{phase_tag}_static", "seasonal")]

        for method_label, ser in [("Dynamic", dyn), ("Static", stat)]:
            for period in [(1980, YEAR_SPLIT), (YEAR_SPLIT + 1, yrs.max())]:
                s, c_lo, c_hi = linear_trend_with_ci(yrs, ser, *period)
                print(
                    f"{label:17s} | {method_label:8s} | "
                    f"{period[0]}–{period[1]}: "
                    f"{s:6.2f} days/dec "
                    f"(95% CI [{c_lo:6.2f}, {c_hi:6.2f}])"
                )

    # -----------------------------------------------------------------
    # Optional: diagnostic figure – seasonal vs all-ice for both methods
    # -----------------------------------------------------------------
    MAKE_DIAGNOSTIC_FIG = True
    if MAKE_DIAGNOSTIC_FIG:
        print("Building seasonal-vs-all diagnostic figure...")
        fig2, axes2 = plt.subplots(3, 1, figsize=(7.5, 6.0), sharex=True)
        fig2.patch.set_facecolor("white")

        for ax, (title, phase_tag) in zip(axes2, panels):
            yrs, dyn_seas = series[(f"{phase_tag}_dynamic", "seasonal")]
            _, dyn_all = series[(f"{phase_tag}_dynamic", "all")]
            _, stat_seas = series[(f"{phase_tag}_static", "seasonal")]
            _, stat_all = series[(f"{phase_tag}_static", "all")]

            # dynamic
            ax.plot(yrs, dyn_seas, color="black", linewidth=1.6,
                    label="Dynamic seasonal")
            ax.plot(yrs, dyn_all, color="black", linewidth=1.0,
                    linestyle="--", alpha=0.6,
                    label="Dynamic all-ice")

            # static
            ax.plot(yrs, stat_seas, color="#1f77b4", linewidth=1.6,
                    label="Static seasonal")
            ax.plot(yrs, stat_all, color="#1f77b4", linewidth=1.0,
                    linestyle="--", alpha=0.6,
                    label="Static all-ice")

            if YEAR_SPLIT in yrs:
                ax.axvline(YEAR_SPLIT, color="red", linestyle="--",
                           linewidth=1.0, alpha=0.7)

            ax.set_ylabel("Days", fontsize=9)
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.grid(alpha=0.25, linewidth=0.5)

        axes2[-1].set_xlabel("Year", fontsize=9)

        # one legend only
        h2, l2 = axes2[0].get_legend_handles_labels()
        axes2[0].legend(h2, l2, loc="upper left", frameon=False, ncol=2)

        fig2.tight_layout()

        outpath2 = get_fig_path(
            project_root=PROJECT_ROOT,
            subfolder=SUBFOLDER,
            fig_name="Fig_FS_MS_seasonal_vs_all_anomaly_timeseries.png",
        )
        save_and_upload(
            fig2,
            outpath2,
            remote_root=REMOTE_ROOT,
            remote_subdir=SUBFOLDER,
        )


if __name__ == "__main__":
    main()
