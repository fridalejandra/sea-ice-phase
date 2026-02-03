#!/usr/bin/env python
# ============================================================
# JJA Monthly SIC Variability (POSTER SAFE)
#
# - Bootstrap SIC
# - June / July / August panels
# - NO xarray plotting for land
# - NO contours
# - Explicit imshow everywhere
# ============================================================

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import ListedColormap

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

# ============================================================
# HELPERS
# ============================================================
def drop_feb29(da):
    return da.sel(time=~((da.time.dt.month == 2) & (da.time.dt.day == 29)))

def ensure_01(sic):
    return sic / 100.0 if float(sic.max()) > 1.5 else sic

def style_panel(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_facecolor("white")

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

    ax.set_aspect("equal")

def draw_continent(ax, land_mask):
    """
    Draw land with pure matplotlib to avoid xarray histogram bugs.
    """
    # ensure 2D
    if land_mask.ndim == 3:
        lm = land_mask.isel(time=0).values
    else:
        lm = land_mask.values

    if not np.any(lm):
        return

    # land fill
    ax.imshow(
        lm.astype(float),
        origin="lower",
        cmap=ListedColormap(["0.65"]),
        zorder=10
    )

    # coastline
    ax.contour(
        lm.astype(float),
        levels=[0.5],
        colors="0.15",
        linewidths=3.0,
        zorder=11
    )

# ============================================================
# LOAD & PREPROCESS (FLAG-SAFE)
# ============================================================
ds = xr.open_dataset(sic_file)

sic_raw = ds[sic_var]

# Land = pixels that are NaN for all times
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
    std_maps[name] = std

# shared color scale
vmax = float(max(m.max() for m in std_maps.values()))

# ============================================================
# PLOT
# ============================================================
fig, axes = plt.subplots(
    1, 3,
    figsize=(14, 5),
    constrained_layout=True
)

for ax, (name, std) in zip(axes, std_maps.items()):

    im = std.plot.imshow(
        ax=ax,
        cmap=cm,
        vmin=0,
        vmax=vmax,
        add_colorbar=False
    )

    draw_continent(ax, land_mask)
    style_panel(ax)
    ax.set_title(name, fontsize=18, fontweight="bold")

# colorbar
cbar = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.02)
cbar.set_label("Std of SIC", fontsize=14)

out = f"{outdir}/std_SIC_JJA_monthly_poster.png"
plt.savefig(out, dpi=450, bbox_inches="tight")
plt.close()

print(f"Saved: {out}")
print("Done.")
