#!/usr/bin/env python
# ============================================================
# JJA Monthly SIC Variability Maps (POSTER STYLE)
#
# - Bootstrap SIC (flag-safe)
# - June / July / August side-by-side
# - Land handled correctly (packed flags)
# - Optional significance contours for Post − Pre
# ============================================================

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import ListedColormap

# ============================================================
# PATHS / VARS
# ============================================================
sic_file = "/user/geog/falejandraperez/sea-ice-phase/data/merged/merged_bootstrap_SH_latest.nc"
sic_var  = "N07_ICECON"

outdir = "/user/geog/falejandraperez/sea-ice-phase/results/figures/clic_poster"

pre_slice  = slice("1979-01-01", "2016-12-31")
post_slice = slice("2017-01-01", "2024-12-31")

MONTHS = {
    "June": 6,
    "July": 7,
    "August": 8,
}

# variability mode
VAR_MODE = "sic"    # "sic" or "dsic"

# plotting
cmap_main = "magma_r"
fill_land = "0.65"
edge_land = "0.15"

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

    circle = Circle((0.5, 0.5), 0.5, transform=ax.transAxes,
                    facecolor="none", edgecolor="none")
    ax.add_patch(circle)

    for artist in ax.get_children():
        try:
            artist.set_clip_path(circle)
        except Exception:
            pass

    ax.set_aspect("equal")

def draw_continent(ax, land_mask):
    if land_mask.sum() == 0:
        return

    land = xr.where(land_mask, 1.0, np.nan)
    land.plot(
        ax=ax,
        cmap=ListedColormap([fill_land]),
        add_colorbar=False,
        zorder=10
    )

    land_mask.plot.contour(
        ax=ax,
        levels=[0.5],
        colors=edge_land,
        linewidths=3.0,
        add_colorbar=False,
        zorder=11
    )

# ============================================================
# LOAD & FLAG-SAFE PREPROCESSING
# ============================================================
ds = xr.open_dataset(sic_file, chunks={"time": 365})
sic_raw = ds[sic_var]

# Bootstrap packed values:
#   valid SIC ∈ [0, 1]
#   land / missing > 1
land_mask  = sic_raw > 1.0
ocean_mask = sic_raw <= 1.0

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
# COMPUTE MONTHLY FIELDS
# ============================================================
std_maps = {}
sig_masks = {}

for name, month in MONTHS.items():

    print(f"Processing {name}...")

    is_month = sic["time"].dt.month == month

    sic_pre  = sic.sel(time=pre_slice).where(is_month, drop=True)
    sic_post = sic.sel(time=post_slice).where(is_month, drop=True)

    if VAR_MODE == "dsic":
        fld_pre  = sic_pre.diff("time")
        fld_post = sic_post.diff("time")
    else:
        fld_pre  = sic_pre
        fld_post = sic_post

    fld_pre  = fld_pre.where(ocean_mask)
    fld_post = fld_post.where(ocean_mask)

    std_pre  = fld_pre.std("time", skipna=True)
    std_post = fld_post.std("time", skipna=True)

    diff = std_post - std_pre

    # significance mask: top 10% absolute change
    thresh = np.nanpercentile(np.abs(diff), 90)
    sig = np.abs(diff) >= thresh

    std_maps[name] = std_post
    sig_masks[name] = sig

# shared color scale
vmax = float(max(m.max() for m in std_maps.values()))

# ============================================================
# PLOT: JUNE / JULY / AUGUST
# ============================================================
fig, axes = plt.subplots(
    1, 3,
    figsize=(14, 5),
    constrained_layout=True
)

for ax, (name, std) in zip(axes, std_maps.items()):

    # --- main field ---
    im = std.plot.imshow(
        ax=ax,
        cmap=cm,
        vmin=0,
        vmax=vmax,
        add_colorbar=False
    )

    # --- significance contours (safe) ---
    sig = sig_masks[name]
    if np.any(sig):
        sig.plot.contour(
            ax=ax,
            levels=[0.5],
            colors="black",
            linewidths=1.2,
            add_colorbar=False,
            zorder=9
        )

    draw_continent(ax, land_mask)
    style_panel(ax)
    ax.set_title(name, fontsize=18, fontweight="bold")

# colorbar
cbar = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.02)
cbar.set_label("Std of SIC", fontsize=14)

out = f"{outdir}/std_SIC_JJA_monthly_with_significance.png"
plt.savefig(out, dpi=450, bbox_inches="tight")
plt.close()

print(f"Saved: {out}")
print("Done.")
