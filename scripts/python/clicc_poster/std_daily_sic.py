#!/usr/bin/env python3
"""
Compute and plot daily sea ice concentration (SIC) variability as the
standard deviation of the magnitude of day-to-day SIC changes: std(|ΔSIC|).

This is a "basics-first" script:
- No SIC threshold masking by default (you can turn it on at the bottom).
- Same computation for each period.
- Robust scaling check (0–1 vs 0–100).
- Feb 29 removed to avoid leap-day artifacts.
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

# Plot controls
cmap_main = "viridis"
cmap_diff = "RdBu_r"

# If None: choose vmax from the data (recommended)
vmax_main = None          # e.g., 0.05 if you want fixed scaling
vmax_diff = None          # e.g., 0.02 if you want fixed scaling

# Quantile-based color scaling (used only when vmax_* is None)
vmax_quantile_main = 0.99
vmax_quantile_diff = 0.99

# Optional masking (OFF for "start with basics")
apply_common_mask = False
mask_threshold = 0.15      # SIC threshold used only if apply_common_mask=True

# Output
out_png = "/user/geog/falejandraperez/sea-ice-phase/results/figures/clic_poster/std_daily_sic_variability.png"
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
    """Remove Feb 29 to avoid leap-day discontinuities."""
    return sic_da.sel(time=~((sic_da.time.dt.month == 2) & (sic_da.time.dt.day == 29)))


def daily_volatility_std(sic_da: xr.DataArray) -> xr.DataArray:
    """
    Compute std(|ΔSIC|) over time for each grid cell.
    ΔSIC is day-to-day difference; abs() gives magnitude (volatility).
    """
    dsic = sic_da.diff("time")
    metric = dsic.abs()
    return metric.std("time", skipna=True)


def qval(da: xr.DataArray, q: float) -> float:
    """Return float quantile (robust with xarray/dask)."""
    return float(da.quantile(q).values)


# -----------------------------
# LOAD + PREP
# -----------------------------
ds = xr.open_dataset(sic_file)
sic = ds[sic_var]

sic = ensure_fractional_sic(sic)
sic = drop_feb29(sic)

# Split periods
sic_pre  = sic.sel(time=slice(pre_start,  pre_end))
sic_post = sic.sel(time=slice(post_start, post_end))

# Compute std(|ΔSIC|)
std_pre  = daily_volatility_std(sic_pre)
std_post = daily_volatility_std(sic_post)

# Optional: common (intersection) mask based on mean SIC > threshold in BOTH periods
if apply_common_mask:
    mask_common = (sic_pre.mean("time") > mask_threshold) & (sic_post.mean("time") > mask_threshold)
    std_pre  = std_pre.where(mask_common)
    std_post = std_post.where(mask_common)

# Difference
std_diff = std_post - std_pre


# -----------------------------
# COLOR LIMITS (DON'T GUESS)
# -----------------------------
# Main panels: use a shared vmax so pre/post are comparable
if vmax_main is None:
    vmax_main = max(qval(std_pre, vmax_quantile_main), qval(std_post, vmax_quantile_main))

# Difference: symmetric scaling around zero
if vmax_diff is None:
    vmax_diff = qval(std_diff.abs(), vmax_quantile_diff)


# -----------------------------
# PLOT
# -----------------------------
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), constrained_layout=True)

# Pre
im0 = std_pre.plot(
    ax=axes[0],
    cmap=cmap_main,
    vmin=0,
    vmax=vmax_main,
    add_colorbar=False
)
axes[0].set_title(f"Std(|ΔSIC|) (Pre: {pre_start[:4]}–{pre_end[:4]})")
axes[0].set_xlabel("")
axes[0].set_ylabel("")

# Post
im1 = std_post.plot(
    ax=axes[1],
    cmap=cmap_main,
    vmin=0,
    vmax=vmax_main,
    add_colorbar=False
)
axes[1].set_title(f"Std(|ΔSIC|) (Post: {post_start[:4]}–{post_end[:4]})")
axes[1].set_xlabel("")
axes[1].set_ylabel("")

# Diff
im2 = std_diff.plot(
    ax=axes[2],
    cmap=cmap_diff,
    vmin=-vmax_diff,
    vmax=+vmax_diff,
    add_colorbar=False
)
axes[2].set_title("Post − Pre difference")
axes[2].set_xlabel("")
axes[2].set_ylabel("")

# Colorbars
cbar1 = fig.colorbar(im0, ax=axes[:2], orientation="vertical", shrink=0.9)
cbar1.set_label("Std of daily SIC change magnitude, std(|ΔSIC|)")

cbar2 = fig.colorbar(im2, ax=axes[2], orientation="vertical", shrink=0.9)
cbar2.set_label("Δ Std(|ΔSIC|)")

# Save
plt.savefig(out_png, dpi=dpi)
plt.close()

print("Saved:", out_png)
print("vmax_main =", vmax_main)
print("vmax_diff =", vmax_diff)
print("apply_common_mask =", apply_common_mask, "mask_threshold =", mask_threshold)
