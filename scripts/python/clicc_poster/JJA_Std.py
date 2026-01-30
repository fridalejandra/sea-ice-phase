#!/usr/bin/env python
# ============================================================
# JJA pre vs post maps (poster style)
# - dark grey continent
# - white for missing ocean
# - round panels, no ticks
# - Std(SIC) (optionally Std(ΔSIC))
# ============================================================

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ============================================================
# PATHS / VARS
# ============================================================
sic_file = "/user/geog/falejandraperez/sea-ice-phase/data/merged/merged_bootstrap_SH_latest.nc"
sic_var  = "N07_ICECON"

outdir = "/user/geog/falejandraperez/sea-ice-phase/results/figures/clic_poster/"

pre_slice  = slice("1979-01-01", "2016-12-31")
post_slice = slice("2017-01-01", "2024-12-31")

# Choose what variability means here:
#   "sic"  -> Std(SIC) across JJA days
#   "dsic" -> Std(ΔSIC) across JJA days
VAR_MODE = "sic"   # change to "dsic" if needed

cmap_main = "magma_r"
cmap_diff = "RdBu_r"

# ============================================================
# HELPERS
# ============================================================
def drop_feb29(da):
    return da.sel(time=~((da.time.dt.month == 2) & (da.time.dt.day == 29)))

def ensure_01(sic):
    # Bootstrap is often 0–100; convert to 0–1 if needed
    return sic / 100.0 if float(sic.max()) > 1.5 else sic

def get_land_mask(ds, sic_da):
    """
    Try to identify a land mask.
    Preference order:
      1) explicit land mask variables (if present)
      2) "always missing" in SIC (fallback)
    Returns boolean mask: True where land/continent.
    """
    candidates = [
        "land", "LAND", "landmask", "LANDMASK",
        "N07_LANDMASK", "mask", "MASK"
    ]
    for v in candidates:
        if v in ds.variables:
            m = ds[v]
            # coerce to boolean
            # common conventions: 1=land, 0=ocean OR vice versa.
            # We'll assume >0.5 means land.
            try:
                return (m > 0.5)
            except Exception:
                pass

    # fallback: land = places that are NA at a reference time
    # (works if continent is encoded as NaN)
    return sic_da.isel(time=0).isnull()

def style_poster_panel(ax, continent_color="0.25"):
    """Poster-style Antarctic panel: round, no ticks, white missing ocean."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_facecolor("white")  # missing ocean is white

    # circular clip
    circle = Circle((0.5, 0.5), 0.5, transform=ax.transAxes,
                    facecolor="none", edgecolor="none")
    ax.add_patch(circle)
    for artist in ax.get_children():
        try:
            artist.set_clip_path(circle)
        except Exception:
            pass

    ax.set_aspect("equal")

def draw_continent(ax, land_mask, color="0.25"):
    """
    Draw continent as a solid overlay.
    land_mask should be boolean with same x/y dims as SIC.
    """
    # plot as an overlay where land=True
    land = xr.where(land_mask, 1.0, np.nan)
    land.plot(
        ax=ax,
        cmap=plt.matplotlib.colors.ListedColormap([color]),
        add_colorbar=False,
        interpolation="nearest"
    )

# ============================================================
# LOAD
# ============================================================
ds = xr.open_dataset(sic_file, chunks={"time": 365})
sic = ensure_01(drop_feb29(ds[sic_var]))

# JJA only
is_jja = sic["time"].dt.month.isin([6, 7, 8])

sic_pre  = sic.sel(time=pre_slice).where(is_jja, drop=True)
sic_post = sic.sel(time=post_slice).where(is_jja, drop=True)

# masks: only keep ocean cells where SIC > 0 (and keep NAs as NA)
mask_pre  = sic_pre  > 0
mask_post = sic_post > 0

# land mask (for continent shading)
land_mask = get_land_mask(ds, sic)

# ============================================================
# COMPUTE FIELD
# ============================================================
if VAR_MODE == "dsic":
    # daily change within each period
    fld_pre  = sic_pre.diff("time")
    fld_post = sic_post.diff("time")
    fld_pre  = fld_pre.where(mask_pre.isel(time=slice(1, None)))
    fld_post = fld_post.where(mask_post.isel(time=slice(1, None)))
    label_main = "Std of daily SIC change"
else:
    fld_pre  = sic_pre.where(mask_pre)
    fld_post = sic_post.where(mask_post)
    label_main = "Std of SIC"

std_pre  = fld_pre.std("time", skipna=True).compute()
std_post = fld_post.std("time", skipna=True).compute()
std_diff = (std_post - std_pre).compute()

vmax      = float(max(std_pre.max(), std_post.max()))
vmax_diff = float(np.abs(std_diff).max())

# make sure NaNs plot as white
cm_main = plt.get_cmap(cmap_main).copy()
cm_main.set_bad("white")
cm_diff = plt.get_cmap(cmap_diff).copy()
cm_diff.set_bad("white")

# ============================================================
# PLOT
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)

# --- main panels
im0 = std_pre.plot(ax=axes[0], cmap=cm_main, vmin=0, vmax=vmax, add_colorbar=False)
im1 = std_post.plot(ax=axes[1], cmap=cm_main, vmin=0, vmax=vmax, add_colorbar=False)
im2 = std_diff.plot(ax=axes[2], cmap=cm_diff, vmin=-vmax_diff, vmax=vmax_diff, add_colorbar=False)

# --- continent overlay (dark grey) + styling
for ax in axes:
    # continent first so it isn't overwritten by pcolormesh edges
    draw_continent(ax, land_mask, color="0.25")
    style_poster_panel(ax)

axes[0].set_title("JJA Std(SIC) Pre",  fontsize=18, fontweight="bold")
axes[1].set_title("JJA Std(SIC) Post", fontsize=18, fontweight="bold")
axes[2].set_title("JJA Post − Pre",    fontsize=18, fontweight="bold")

# --- colorbars
cbar1 = fig.colorbar(im0, ax=axes[:2], shrink=0.9, pad=0.02)
cbar1.set_label(label_main, fontsize=14)

cbar2 = fig.colorbar(im2, ax=axes[2], shrink=0.9, pad=0.02)
cbar2.set_label("Post − Pre", fontsize=14)

# ============================================================
# SAVE
# ============================================================
mode_tag = "std_dSIC" if VAR_MODE == "dsic" else "std_SIC"
out = f"{outdir}/{mode_tag}_JJA_pre_post_diff_poster.png"
plt.savefig(out, dpi=450, bbox_inches="tight")
plt.close()

print(f"Saved: {out}")
