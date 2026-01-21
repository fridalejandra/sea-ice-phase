#!/usr/bin/env python3
"""
Eulerian daily SIC variability diagnostics with:
- common day-of-year climatology (computed from full record)
- process-based seasons (month groupings)
- outputs pre/post/diff triptychs for each metric and season

Metrics:
1) std(|ΔSIC|)  : std of magnitude of daily SIC changes
2) std(SIC)     : std of SIC (includes seasonal + subseasonal)
3) std(SIC')    : std of SIC anomalies after removing common DOY climatology
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# USER SETTINGS
# -----------------------------
sic_file = "/user/geog/falejandraperez/sea-ice-phase/data/merged/merged_bootstrap_SH_latest.nc"
sic_var  = "N07_ICECON"

pre_start, pre_end   = "1979-01-01", "2016-12-31"
post_start, post_end = "2017-01-01", "2024-12-31"

out_dir = Path("/user/geog/falejandraperez/sea-ice-phase/results/figures/clic_poster/seasonal_commonclim")
out_dir.mkdir(parents=True, exist_ok=True)

dpi = 300
cmap_main = "viridis"
cmap_diff = "RdBu_r"
main_q = 0.99
diff_q = 0.99

# Process-based seasons (edit as you like)
# These are simple month bins; keep them fixed for transparency.
SEASONS = {
    "Advance_MarJun":  [3, 4, 5, 6],
    "Winter_JulSep":   [7, 8, 9],
    "Retreat_OctJan":  [10, 11, 12, 1],
    "LateSummer_Feb":  [2],   # optional; keeps Feb separate
}

# -----------------------------
# HELPERS
# -----------------------------
def ensure_fractional_sic(sic_da: xr.DataArray) -> xr.DataArray:
    mx = float(sic_da.max().values)
    return sic_da / 100.0 if mx > 1.5 else sic_da

def drop_feb29(sic_da: xr.DataArray) -> xr.DataArray:
    return sic_da.sel(time=~((sic_da.time.dt.month == 2) & (sic_da.time.dt.day == 29)))

def qval(da: xr.DataArray, q: float) -> float:
    return float(da.quantile(q).values)

def subset_months(da: xr.DataArray, months: list[int]) -> xr.DataArray:
    return da.sel(time=da.time.dt.month.isin(months))

def common_doy_climatology(sic_full: xr.DataArray) -> xr.DataArray:
    """
    Compute common DOY climatology from full record (after Feb29 removal).
    Returns DataArray with dimension dayofyear (1..365) + spatial dims.
    """
    doy = sic_full.time.dt.dayofyear
    return sic_full.groupby(doy).mean("time", skipna=True)

def remove_common_seasonal_cycle(sic_da: xr.DataArray, clim_doy: xr.DataArray) -> xr.DataArray:
    """
    Subtract the common DOY climatology.
    """
    doy = sic_da.time.dt.dayofyear
    # align by doy index
    return sic_da.groupby(doy) - clim_doy

def metric_std_abs_dSIC(sic_da: xr.DataArray) -> xr.DataArray:
    dsic = sic_da.diff("time")
    return np.abs(dsic).std("time", skipna=True)

def metric_std_SIC(sic_da: xr.DataArray) -> xr.DataArray:
    return sic_da.std("time", skipna=True)

def metric_std_SIC_anom(sic_anom_da: xr.DataArray) -> xr.DataArray:
    return sic_anom_da.std("time", skipna=True)

def plot_triptych(field_pre, field_post, title_left, title_mid, title_right,
                  cbar_label_main, cbar_label_diff, outfile_png):
    field_diff = field_post - field_pre

    vmax_main = max(qval(field_pre, main_q), qval(field_post, main_q))
    vmax_diff = qval(np.abs(field_diff), diff_q)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    im0 = field_pre.plot(ax=axes[0], cmap=cmap_main, vmin=0, vmax=vmax_main, add_colorbar=False)
    axes[0].set_title(title_left); axes[0].set_xlabel(""); axes[0].set_ylabel("")

    im1 = field_post.plot(ax=axes[1], cmap=cmap_main, vmin=0, vmax=vmax_main, add_colorbar=False)
    axes[1].set_title(title_mid); axes[1].set_xlabel(""); axes[1].set_ylabel("")

    im2 = field_diff.plot(ax=axes[2], cmap=cmap_diff, vmin=-vmax_diff, vmax=vmax_diff, add_colorbar=False)
    axes[2].set_title(title_right); axes[2].set_xlabel(""); axes[2].set_ylabel("")

    cbar1 = fig.colorbar(im0, ax=axes[:2], orientation="vertical", shrink=0.9)
    cbar1.set_label(cbar_label_main)

    cbar2 = fig.colorbar(im2, ax=axes[2], orientation="vertical", shrink=0.9)
    cbar2.set_label(cbar_label_diff)

    plt.savefig(outfile_png, dpi=dpi)
    plt.close()

    print("Saved:", outfile_png)
    print("  vmax_main =", vmax_main, "vmax_diff =", vmax_diff)

# -----------------------------
# LOAD + COMMON CLIMATOLOGY
# -----------------------------
ds = xr.open_dataset(sic_file)
sic = ensure_fractional_sic(ds[sic_var])
sic = drop_feb29(sic)

# Split periods
sic_pre  = sic.sel(time=slice(pre_start,  pre_end))
sic_post = sic.sel(time=slice(post_start, post_end))

# Common DOY climatology from FULL record (recommended for change questions)
clim_doy = common_doy_climatology(sic)

# Anomalies relative to common climatology
sic_pre_anom  = remove_common_seasonal_cycle(sic_pre,  clim_doy)
sic_post_anom = remove_common_seasonal_cycle(sic_post, clim_doy)

# -----------------------------
# LOOP OVER PROCESS-BASED SEASONS
# -----------------------------
for season_name, months in SEASONS.items():
    # subset months for each field
    pre_sic  = subset_months(sic_pre, months)
    post_sic = subset_months(sic_post, months)

    pre_anom  = subset_months(sic_pre_anom, months)
    post_anom = subset_months(sic_post_anom, months)

    # --- Metric 1: std(|ΔSIC|)
    pre_m1  = metric_std_abs_dSIC(pre_sic)
    post_m1 = metric_std_abs_dSIC(post_sic)
    plot_triptych(
        pre_m1, post_m1,
        f"Std(|ΔSIC|) {season_name} (Pre)",
        f"Std(|ΔSIC|) {season_name} (Post)",
        "Post − Pre difference",
        "Std of |daily SIC change|, std(|ΔSIC|)",
        "Δ std(|ΔSIC|)",
        out_dir / f"{season_name}_fig1_std_abs_dSIC.png"
    )

    # --- Metric 2: std(SIC)
    pre_m2  = metric_std_SIC(pre_sic)
    post_m2 = metric_std_SIC(post_sic)
    plot_triptych(
        pre_m2, post_m2,
        f"Std(SIC) {season_name} (Pre)",
        f"Std(SIC) {season_name} (Post)",
        "Post − Pre difference",
        "Std of SIC, std(SIC)",
        "Δ std(SIC)",
        out_dir / f"{season_name}_fig2_std_SIC.png"
    )

    # --- Metric 3: std(SIC') using common climatology
    pre_m3  = metric_std_SIC_anom(pre_anom)
    post_m3 = metric_std_SIC_anom(post_anom)
    plot_triptych(
        pre_m3, post_m3,
        f"Std(SIC') {season_name} (Pre)",
        f"Std(SIC') {season_name} (Post)",
        "Post − Pre difference",
        "Std of SIC anomalies (common DOY climatology removed), std(SIC')",
        "Δ std(SIC')",
        out_dir / f"{season_name}_fig3_std_SIC_anom_commonclim.png"
    )
