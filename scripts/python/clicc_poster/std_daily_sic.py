#!/usr/bin/env python3
"""
Three diagnostics of daily sea ice concentration (SIC) variability:

FIG 1: std(|ΔSIC|)  = standard deviation of the magnitude of day-to-day SIC changes
FIG 2: std(SIC)     = standard deviation of SIC (includes seasonal cycle)
FIG 3: std(SIC')    = standard deviation of SIC anomalies after removing day-of-year climatology

Each figure is 3 panels: Pre, Post, Post-Pre.

Notes:
- Removes Feb 29 to keep day-of-year climatology consistent (365-day).
- Uses quantile-based color scaling so you don't guess vmax.
- Uses numpy abs() for compatibility with xarray versions lacking DataArray.abs().
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# USER SETTINGS
# -----------------------------
sic_file = "/user/geog/falejandraperez/sea-ice-phase/data/merged/merged_bootstrap_SH_latest.nc"
sic_var  = "N07_ICECON"

pre_start, pre_end   = "1979-01-01", "2016-12-31"
post_start, post_end = "2017-01-01", "2024-12-31"

out_dir = "/user/geog/falejandraperez/sea-ice-phase/results/figures/clic_poster"

# Color scaling via quantiles (recommended)
main_q = 0.99   # for pre/post panels
diff_q = 0.99   # for diff panel (symmetric about 0)

# Colormaps
cmap_main = "viridis"
cmap_diff = "RdBu_r"

dpi = 300


# -----------------------------
# HELPERS
# -----------------------------
def ensure_fractional_sic(sic_da: xr.DataArray) -> xr.DataArray:
    """Ensure SIC is in 0–1 units."""
    mx = float(sic_da.max().values)
    if mx > 1.5:
        return sic_da / 100.0
    return sic_da


def drop_feb29(sic_da: xr.DataArray) -> xr.DataArray:
    """Remove Feb 29 to keep day-of-year climatology consistent."""
    return sic_da.sel(time=~((sic_da.time.dt.month == 2) & (sic_da.time.dt.day == 29)))


def qval(da: xr.DataArray, q: float) -> float:
    """Return float quantile robustly."""
    return float(da.quantile(q).values)


def std_abs_daily_change(sic_da: xr.DataArray) -> xr.DataArray:
    """std(|ΔSIC|) over time, computed at each grid cell."""
    dsic = sic_da.diff("time")
    metric = np.abs(dsic)
    return metric.std("time", skipna=True)


def std_sic(sic_da: xr.DataArray) -> xr.DataArray:
    """std(SIC) over time, computed at each grid cell."""
    return sic_da.std("time", skipna=True)


def std_sic_anom_doy(sic_da: xr.DataArray) -> xr.DataArray:
    """
    std(SIC') where SIC' = SIC - climatology(day-of-year).
    Day-of-year climatology is computed from the same period.
    """
    doy = sic_da.time.dt.dayofyear
    clim = sic_da.groupby(doy).mean("time", skipna=True)
    anom = sic_da.groupby(doy) - clim
    return anom.std("time", skipna=True)


def plot_triptych(field_pre, field_post, title_pre, title_post, title_diff,
                  cbar_label_main, cbar_label_diff, outfile_png):
    """Make a 3-panel plot: pre, post, post-pre."""
    field_diff = field_post - field_pre

    # Shared scaling for pre/post
    vmax_main = max(qval(field_pre, main_q), qval(field_post, main_q))
    # Symmetric scaling for diff
    vmax_diff = qval(np.abs(field_diff), diff_q)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), constrained_layout=True)

    im0 = field_pre.plot(ax=axes[0], cmap=cmap_main, vmin=0, vmax=vmax_main, add_colorbar=False)
    axes[0].set_title(title_pre)
    axes[0].set_xlabel(""); axes[0].set_ylabel("")

    im1 = field_post.plot(ax=axes[1], cmap=cmap_main, vmin=0, vmax=vmax_main, add_colorbar=False)
    axes[1].set_title(title_post)
    axes[1].set_xlabel(""); axes[1].set_ylabel("")

    im2 = field_diff.plot(ax=axes[2], cmap=cmap_diff, vmin=-vmax_diff, vmax=vmax_diff, add_colorbar=False)
    axes[2].set_title(title_diff)
    axes[2].set_xlabel(""); axes[2].set_ylabel("")

    cbar1 = fig.colorbar(im0, ax=axes[:2], orientation="vertical", shrink=0.9)
    cbar1.set_label(cbar_label_main)

    cbar2 = fig.colorbar(im2, ax=axes[2], orientation="vertical", shrink=0.9)
    cbar2.set_label(cbar_label_diff)

    plt.savefig(outfile_png, dpi=dpi)
    plt.close()

    print("Saved:", outfile_png)
    print("  vmax_main =", vmax_main)
    print("  vmax_diff =", vmax_diff)


# -----------------------------
# LOAD + PREP
# -----------------------------
ds = xr.open_dataset(sic_file)
sic = ds[sic_var]
sic = ensure_fractional_sic(sic)
sic = drop_feb29(sic)

sic_pre  = sic.sel(time=slice(pre_start,  pre_end))
sic_post = sic.sel(time=slice(post_start, post_end))


# -----------------------------
# FIGURE 1: std(|ΔSIC|)
# -----------------------------
std_absdsic_pre  = std_abs_daily_change(sic_pre)
std_absdsic_post = std_abs_daily_change(sic_post)

plot_triptych(
    std_absdsic_pre,
    std_absdsic_post,
    f"Std(|ΔSIC|) (Pre: {pre_start[:4]}–{pre_end[:4]})",
    f"Std(|ΔSIC|) (Post: {post_start[:4]}–{post_end[:4]})",
    "Post − Pre difference",
    "Std of |daily SIC change|, std(|ΔSIC|)",
    "Δ std(|ΔSIC|)",
    f"{out_dir}/fig1_std_abs_daily_change.png"
)


# -----------------------------
# FIGURE 2: std(SIC)
# -----------------------------
std_sic_pre  = std_sic(sic_pre)
std_sic_post = std_sic(sic_post)

plot_triptych(
    std_sic_pre,
    std_sic_post,
    f"Std(SIC) (Pre: {pre_start[:4]}–{pre_end[:4]})",
    f"Std(SIC) (Post: {post_start[:4]}–{post_end[:4]})",
    "Post − Pre difference",
    "Std of SIC, std(SIC)",
    "Δ std(SIC)",
    f"{out_dir}/fig2_std_sic.png"
)


# -----------------------------
# FIGURE 3: std(SIC anomalies), seasonal cycle removed
# -----------------------------
std_anom_pre  = std_sic_anom_doy(sic_pre)
std_anom_post = std_sic_anom_doy(sic_post)

plot_triptych(
    std_anom_pre,
    std_anom_post,
    f"Std(SIC') (Pre: {pre_start[:4]}–{pre_end[:4]})",
    f"Std(SIC') (Post: {post_start[:4]}–{post_end[:4]})",
    "Post − Pre difference",
    "Std of SIC anomalies (DOY climatology removed), std(SIC')",
    "Δ std(SIC')",
    f"{out_dir}/fig3_std_sic_anom_doy.png"
)
