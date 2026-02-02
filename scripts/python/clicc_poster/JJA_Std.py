#!/usr/bin/env python
# ============================================================
# Seasonal pre vs post maps (POSTER STYLE)
# - dark grey continent
# - white missing ocean
# - round panels, no ticks
# - Std(SIC) or Std(ΔSIC)
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

outdir = "/user/geog/falejandraperez/sea-ice-phase/results/figures/clic_poster/"

pre_slice  = slice("1979-01-01", "2016-12-31")
post_slice = slice("2017-01-01", "2024-12-31")

# choose variability definition
VAR_MODE = "dsic"   # "sic" or "dsic"

cmap_main = "magma_r"
cmap_diff = "RdBu_r"

seasons = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
}

# ============================================================
# HELPERS
# ============================================================
def drop_feb29(da):
    return da.sel(time=~((da.time.dt.month == 2) & (da.time.dt.day == 29)))

def ensure_01(sic):
    return sic / 100.0 if float(sic.max()) > 1.5 else sic

def get_land_mask(ds, sic_da):
    candidates = [
        "land", "LAND", "landmask", "LANDMASK",
        "N07_LANDMASK", "mask", "MASK"
    ]
    for v in candidates:
        if v in ds.variables:
            return ds[v] > 0.5
    return sic_da.isel(time=0).isnull()

def style_poster_panel(ax):
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

def draw_continent(
    ax,
    land_mask,
    fill_color="0.9",
    edge_color="0.2",
    lw=2.5
):
    # ---- Fill land ----
    land_fill = xr.where(land_mask, 1.0, np.nan)
    land_fill.plot(
        ax=ax,
        cmap=ListedColormap([fill_color]),
        add_colorbar=False,
        zorder=5
    )

    # ---- Draw coastline (explicit boundary) ----
    land_mask.plot.contour(
        ax=ax,
        levels=[0.5],
        colors=edge_color,
        linewidths=lw,
        add_colorbar=False,
        zorder=6
    )


# ============================================================
# LOAD
# ============================================================
ds = xr.open_dataset(sic_file, chunks={"time": 365})
sic = ensure_01(drop_feb29(ds[sic_var]))

land_mask = get_land_mask(ds, sic)

# colormaps with white NaNs
cm_main = plt.get_cmap(cmap_main).copy()
cm_main.set_bad("white")
cm_diff = plt.get_cmap(cmap_diff).copy()
cm_diff.set_bad("white")

# ============================================================
# LOOP SEASONS
# ============================================================
for sname, months in seasons.items():

    print(f"Processing {sname}...")

    is_season = sic["time"].dt.month.isin(months)

    sic_pre  = sic.sel(time=pre_slice).where(is_season, drop=True)
    sic_post = sic.sel(time=post_slice).where(is_season, drop=True)

    mask_pre  = sic_pre  > 0
    mask_post = sic_post > 0

    # --------------------------
    # FIELD DEFINITION
    # --------------------------
    if VAR_MODE == "dsic":
        fld_pre  = sic_pre.diff("time")
        fld_post = sic_post.diff("time")

        fld_pre  = fld_pre.where(mask_pre.isel(time=slice(1, None)))
        fld_post = fld_post.where(mask_post.isel(time=slice(1, None)))

        label_main = "Std of daily SIC change"
        mode_tag = "std_dSIC"
    else:
        fld_pre  = sic_pre.where(mask_pre)
        fld_post = sic_post.where(mask_post)

        label_main = "Std of SIC"
        mode_tag = "std_SIC"

    std_pre  = fld_pre.std("time", skipna=True).compute()
    std_post = fld_post.std("time", skipna=True).compute()
    std_diff = (std_post - std_pre).compute()

    vmax      = float(max(std_pre.max(), std_post.max()))
    vmax_diff = float(np.abs(std_diff).max())

    # ======================================================
    # PLOT
    # ======================================================
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)

    im0 = std_pre.plot(ax=axes[0], cmap=cm_main, vmin=0, vmax=vmax, add_colorbar=False)
    im1 = std_post.plot(ax=axes[1], cmap=cm_main, vmin=0, vmax=vmax, add_colorbar=False)
    im2 = std_diff.plot(ax=axes[2], cmap=cm_diff,
                        vmin=-vmax_diff, vmax=vmax_diff,
                        add_colorbar=False)

    for ax in axes:
        draw_continent(
            ax,
            land_mask,
            fill_color="0.9",
            edge_color="0.2",
            lw=2.5
        )
        style_poster_panel(ax)

    axes[0].set_title(f"{sname} Pre",  fontsize=18, fontweight="bold")
    axes[1].set_title(f"{sname} Post", fontsize=18, fontweight="bold")
    axes[2].set_title(f"{sname} Post − Pre", fontsize=18, fontweight="bold")

    cbar1 = fig.colorbar(im0, ax=axes[:2], shrink=0.9, pad=0.02)
    cbar1.set_label(label_main, fontsize=14)

    cbar2 = fig.colorbar(im2, ax=axes[2], shrink=0.9, pad=0.02)
    cbar2.set_label("Post − Pre", fontsize=14)

    out = f"{outdir}/{mode_tag}_{sname}_pre_post_diff_poster.png"
    plt.savefig(out, dpi=450, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out}")

print("Done.")
