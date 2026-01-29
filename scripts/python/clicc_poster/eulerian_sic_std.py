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

def poster_ax(ax):
    """Poster-style Antarctic panel."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_facecolor("0.85")  # grey continent/background
    ax.set_aspect("equal")

    # circular clip
    circle = Circle((0.5, 0.5), 0.5, transform=ax.transAxes,
                    facecolor="none", edgecolor="none")
    ax.add_patch(circle)
    for artist in ax.get_children():
        artist.set_clip_path(circle)

# ============================================================
# LOAD
# ============================================================
ds = xr.open_dataset(sic_file, chunks={"time": 365})
sic = ensure_01(drop_feb29(ds[sic_var]))

sic_pre  = sic.sel(time=pre_slice)
sic_post = sic.sel(time=post_slice)

dsic_pre  = sic_pre.diff("time")
dsic_post = sic_post.diff("time")

month = sic["time"].dt.month
season_mask = {s: month.isin(m) for s, m in seasons.items()}

# ============================================================
# LOOP SEASONS
# ============================================================
for sname in seasons:

    print(f"Processing {sname}...")

    sp  = sic_pre.where(season_mask[sname], drop=True)
    so  = sic_post.where(season_mask[sname], drop=True)

    dsp = dsic_pre.where(season_mask[sname], drop=True)
    dso = dsic_post.where(season_mask[sname], drop=True)

    # ======================================================
    # FIG A — Std(ΔSIC)  (mask only 0 / NaN)
    # ======================================================
    mask_pre  = sp > 0
    mask_post = so > 0

    std_pre  = dsp.where(mask_pre).std("time", skipna=True).compute()
    std_post = dso.where(mask_post).std("time", skipna=True).compute()
    std_diff = (std_post - std_pre).compute()

    vmax = float(max(std_pre.max(), std_post.max()))
    vmax_diff = float(np.abs(std_diff).max())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im0 = std_pre.plot(ax=axes[0], cmap=cmap_main, vmin=0, vmax=vmax, add_colorbar=False)
    im1 = std_post.plot(ax=axes[1], cmap=cmap_main, vmin=0, vmax=vmax, add_colorbar=False)
    im2 = std_diff.plot(ax=axes[2], cmap=cmap_diff,
                        vmin=-vmax_diff, vmax=vmax_diff,
                        add_colorbar=False)

    for ax in axes:
        poster_ax(ax)

    cbar1 = fig.colorbar(im0, ax=axes[:2], shrink=0.8)
    cbar1.set_label("Std of daily SIC change")

    cbar2 = fig.colorbar(im2, ax=axes[2], shrink=0.8)
    cbar2.set_label("Post − Pre")

    plt.savefig(f"{outdir}/std_dSIC_{sname}_poster.png", dpi=400, bbox_inches="tight")
    plt.close()

    # ======================================================
    # FIG B — Std(SIC)  (NO 15% threshold, ever)
    # ======================================================
    std_pre  = sp.where(mask_pre).std("time", skipna=True).compute()
    std_post = so.where(mask_post).std("time", skipna=True).compute()
    std_diff = (std_post - std_pre).compute()

    vmax = float(max(std_pre.max(), std_post.max()))
    vmax_diff = float(np.abs(std_diff).max())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im0 = std_pre.plot(ax=axes[0], cmap=cmap_main, vmin=0, vmax=vmax, add_colorbar=False)
    im1 = std_post.plot(ax=axes[1], cmap=cmap_main, vmin=0, vmax=vmax, add_colorbar=False)
    im2 = std_diff.plot(ax=axes[2], cmap=cmap_diff,
                        vmin=-vmax_diff, vmax=vmax_diff,
                        add_colorbar=False)

    for ax in axes:
        poster_ax(ax)

    cbar1 = fig.colorbar(im0, ax=axes[:2], shrink=0.8)
    cbar1.set_label("Std of SIC")

    cbar2 = fig.colorbar(im2, ax=axes[2], shrink=0.8)
    cbar2.set_label("Post − Pre")

    plt.savefig(f"{outdir}/std_SIC_{sname}_poster.png", dpi=400, bbox_inches="tight")
    plt.close()

print("Done.")
