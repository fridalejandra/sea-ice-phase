#!/usr/bin/env python
# ============================================================
# JJA Monthly SIC Variability (FINAL POSTER VERSION)
#
# - Bootstrap SIC
# - June / July / August panels
# - Zero values clipped
# - Continent drawn with Cartopy (vector land)
# - Matplotlib imshow only for data (robust)
# ============================================================

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import ListedColormap

import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ============================================================
# PATHS / SETTINGS
# ============================================================
sic_file = "/user/geog/falejandraperez/sea-ice-phase/data/merged/merged_bootstrap_SH_latest.nc"
sic_var  = "N07_ICECON"

outdir = "/user/geog/falejandraperez/sea-ice-phase/results/figures/clic_poster"

MONTHS = {
    "June": 6,
    "July": 7,
    "August": 8,
}

cmap_main = "magma_r"
ZERO_CLIP = 1e-3   # threshold to remove background noise

# ============================================================
# HELPERS
# ============================================================
def drop_feb29(da):
    return da.sel(time=~((da.time.dt.month == 2) & (da.time.dt.day == 29)))

def ensure_01(sic):
    return sic / 100.0 if float(sic.max()) > 1.5 else sic

def style_panel(ax):
    ax.axis("off")
    ax.set_extent([-180, 180, -90, -50], crs=ccrs.PlateCarree())

    # circular clip
    circle = Circle(
        (0.5, 0.5), 0.5,
        transform=ax.transAxes,
        facecolor="none",
        edgecolor="none"
    )
    ax.add_patch(circle)

    for artist in ax.get_children():
        try:
            artist.set_clip_path(circle)
        except Exception:
            pass

# ============================================================
# LOAD & PREPROCESS (FLAG-SAFE)
# ============================================================
ds = xr.open_dataset(sic_file)

sic_raw = ds[sic_var]

# Land = permanently missing pixels
land_mask = sic_raw.isnull().all("time")

# Ocean = ever valid
ocean_mask = ~land_mask

sic = sic_raw.where(ocean_mask)
sic = ensure_01(sic)
sic = drop_feb29(sic)

print("Sanity check:")
print("  land pixels:", int(land_mask.sum()))
print("  SIC min/max:", float(sic.min()), float(sic.max()))

# ============================================================
# COLORMAP
# ============================================================
cm = plt.get_cmap(cmap_main).copy()
cm.set_bad("white")

# ============================================================
# COMPUTE MONTHLY STD MAPS
# ============================================================
std_maps = {}

for name, month in MONTHS.items():
    print(f"Processing {name}...")

    sel = sic["time"].dt.month == month
    sic_m = sic.where(sel, drop=True)

    std = sic_m.std("time", skipna=True)

    # clip near-zero background
    std = std.where(std > ZERO_CLIP)

    std_maps[name] = std

# shared color scale
vmax = float(max(m.max() for m in std_maps.values()))

# ============================================================
# PLOT
# ============================================================
proj = ccrs.SouthPolarStereo()

fig, axes = plt.subplots(
    1, 3,
    figsize=(14, 5),
    constrained_layout=True,
    subplot_kw=dict(projection=proj)
)

for ax, (name, std) in zip(axes, std_maps.items()):

    im = ax.imshow(
        std.values,
        origin="lower",
        cmap=cm,
        vmin=0,
        vmax=vmax,
        transform=proj,
        zorder=1
    )

    # draw continent (vector)
    ax.add_feature(
        cfeature.LAND,
        facecolor="0.65",
        edgecolor="0.15",
        linewidth=2.5,
        zorder=10
    )

    style_panel(ax)
    ax.set_title(name, fontsize=18, fontweight="bold")

# colorbar
cbar = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.02)
cbar.set_label("Std of SIC", fontsize=14)

out = f"{outdir}/std_SIC_JJA_monthly_cartopy_poster.png"
plt.savefig(out, dpi=450, bbox_inches="tight")
plt.close()

print(f"Saved: {out}")
print("Done.")
