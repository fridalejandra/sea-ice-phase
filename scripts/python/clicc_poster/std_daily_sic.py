import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# USER INPUTS
# -----------------------------
sic_file = "/user/geog/falejandraperez/sea-ice-phase/data/merged/merged_bootstrap_SH_latest.nc"
sic_var  = "N07_ICECON"                     # variable name in NetCDF
mask_threshold = 0.15                # SIC threshold
vmax = 0.15                          # adjust after first look
cmap = "viridis"

# -----------------------------
# LOAD DATA
# -----------------------------
ds = xr.open_dataset(sic_file)
sic = ds[sic_var]

# Ensure SIC is 0–1
if sic.max() > 1.5:
    sic = sic / 100.0

# Drop Feb 29 to avoid diff artifacts
sic = sic.sel(time=~((sic.time.dt.month == 2) & (sic.time.dt.day == 29)))

# -----------------------------
# SPLIT PERIODS
# -----------------------------
sic_pre  = sic.sel(time=slice("1979-01-01", "2016-12-31"))
sic_post = sic.sel(time=slice("2017-01-01", "2024-12-31"))

# -----------------------------
# DAILY CHANGES
# -----------------------------
dsic_pre  = sic_pre.diff("time")
dsic_post = sic_post.diff("time")

# -----------------------------
# STANDARD DEVIATION MAPS
# -----------------------------
std_pre  = dsic_pre.std("time", skipna=True)
std_post = dsic_post.std("time", skipna=True)

# -----------------------------
# PERIOD-SPECIFIC MASKS
# -----------------------------
mask_pre  = sic_pre.mean("time")  > mask_threshold
mask_post = sic_post.mean("time") > mask_threshold

std_pre  = std_pre.where(mask_pre)
std_post = std_post.where(mask_post)

# -----------------------------
# DIFFERENCE MAP (OPTIONAL)
# -----------------------------
std_diff = std_post - std_pre

# -----------------------------
# PLOTTING
# -----------------------------
fig, axes = plt.subplots(
    nrows=1, ncols=3, figsize=(15, 5),
    constrained_layout=True
)

# Pre-2017
im0 = std_pre.plot(
    ax=axes[0],
    cmap=cmap,
    vmin=0,
    vmax=vmax,
    add_colorbar=False
)
axes[0].set_title("Daily SIC variability (1979–2016)")
axes[0].set_xlabel("")
axes[0].set_ylabel("")

# Post-2016
im1 = std_post.plot(
    ax=axes[1],
    cmap=cmap,
    vmin=0,
    vmax=vmax,
    add_colorbar=False
)
axes[1].set_title("Daily SIC variability (2017–2024)")
axes[1].set_xlabel("")
axes[1].set_ylabel("")

# Difference
im2 = std_diff.plot(
    ax=axes[2],
    cmap="RdBu_r",
    vmin=-vmax/2,
    vmax=vmax/2,
    add_colorbar=False
)
axes[2].set_title("Post − Pre difference")
axes[2].set_xlabel("")
axes[2].set_ylabel("")

# Shared colorbars
cbar1 = fig.colorbar(im0, ax=axes[:2], orientation="vertical", shrink=0.9)
cbar1.set_label("Std of daily SIC change")

cbar2 = fig.colorbar(im2, ax=axes[2], orientation="vertical", shrink=0.9)
cbar2.set_label("Δ Std of daily SIC change")

plt.savefig('/user/geog/falejandraperez/sea-ice-phase/results/figures/clic_poster/std_daily_sic_variability.png', dpi=300)
