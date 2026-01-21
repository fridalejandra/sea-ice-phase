import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PATHS / VARS
# ============================================================
sic_file = "/user/geog/falejandraperez/sea-ice-phase/data/merged/merged_bootstrap_SH_latest.nc"
sic_var  = "N07_ICECON"

outdir = "/user/geog/falejandraperez/sea-ice-phase/results/figures/clicc_poster/"

# periods
pre_slice  = slice("1979-01-01", "2016-12-31")
post_slice = slice("2017-01-01", "2024-12-31")

# masking threshold
thr_open = 0.15

# plotting
cmap_main = "magma_r"
cmap_diff = "RdBu_r"

# seasons
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

# ============================================================
# LOAD (WITH CHUNKING)
# ============================================================
ds = xr.open_dataset(
    sic_file,
    chunks={"time": 365}
)

sic = ensure_01(drop_feb29(ds[sic_var]))

sic_pre  = sic.sel(time=pre_slice)
sic_post = sic.sel(time=post_slice)

# daily changes — computed ONCE
dsic_pre  = sic_pre.diff("time")
dsic_post = sic_post.diff("time")

# ============================================================
# SEASON MASKS (ONCE)
# ============================================================
month = sic["time"].dt.month
season_mask = {
    s: month.isin(m) for s, m in seasons.items()
}

# ============================================================
# LOOP SEASONS
# ============================================================
for sname in seasons:

    print(f"Processing {sname}...")

    # --------------------------
    # SEASONAL SUBSETS
    # --------------------------
    sp  = sic_pre.where(season_mask[sname], drop=True)
    so  = sic_post.where(season_mask[sname], drop=True)

    dsp = dsic_pre.where(season_mask[sname], drop=True)
    dso = dsic_post.where(season_mask[sname], drop=True)

    # ======================================================
    # FIG A — Std(ΔSIC)
    # ======================================================
    std_pre  = dsp.std("time", skipna=True).compute()
    std_post = dso.std("time", skipna=True).compute()
    std_diff = (std_post - std_pre).compute()

    vmax = float(max(std_pre.max(), std_post.max()))
    vmax_diff = float(np.abs(std_diff).max())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    im0 = std_pre.plot(ax=axes[0], cmap=cmap_main, vmin=0, vmax=vmax, add_colorbar=False)
    axes[0].set_title(f"{sname} Std(ΔSIC) Pre")
    axes[0].set_xlabel(""); axes[0].set_ylabel("")

    im1 = std_post.plot(ax=axes[1], cmap=cmap_main, vmin=0, vmax=vmax, add_colorbar=False)
    axes[1].set_title(f"{sname} Std(ΔSIC) Post")
    axes[1].set_xlabel(""); axes[1].set_ylabel("")

    im2 = std_diff.plot(ax=axes[2], cmap=cmap_diff,
                        vmin=-vmax_diff, vmax=vmax_diff,
                        add_colorbar=False)
    axes[2].set_title(f"{sname} Post − Pre")
    axes[2].set_xlabel(""); axes[2].set_ylabel("")

    cbar1 = fig.colorbar(im0, ax=axes[:2], shrink=0.9)
    cbar1.set_label("Std of daily SIC change")

    cbar2 = fig.colorbar(im2, ax=axes[2], shrink=0.9)
    cbar2.set_label("Δ Std(ΔSIC)")

    plt.savefig(f"{outdir}/std_dSIC_{sname}_pre_post_diff.png", dpi=300)
    plt.close()

    # ======================================================
    # FIG B — Std(SIC), masked BEFORE std
    # ======================================================
    mask_pre  = sp.mean("time", skipna=True) > thr_open
    mask_post = so.mean("time", skipna=True) > thr_open

    sp_m = sp.where(mask_pre)
    so_m = so.where(mask_post)

    std_pre_m  = sp_m.std("time", skipna=True).compute()
    std_post_m = so_m.std("time", skipna=True).compute()
    std_diff_m = (std_post_m - std_pre_m).compute()

    vmax = float(max(std_pre_m.max(), std_post_m.max()))
    vmax_diff = float(np.abs(std_diff_m).max())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    im0 = std_pre_m.plot(ax=axes[0], cmap=cmap_main, vmin=0, vmax=vmax, add_colorbar=False)
    axes[0].set_title(f"{sname} Std(SIC) Pre (SIC>{thr_open})")
    axes[0].set_xlabel(""); axes[0].set_ylabel("")

    im1 = std_post_m.plot(ax=axes[1], cmap=cmap_main, vmin=0, vmax=vmax, add_colorbar=False)
    axes[1].set_title(f"{sname} Std(SIC) Post (SIC>{thr_open})")
    axes[1].set_xlabel(""); axes[1].set_ylabel("")

    im2 = std_diff_m.plot(ax=axes[2], cmap=cmap_diff,
                          vmin=-vmax_diff, vmax=vmax_diff,
                          add_colorbar=False)
    axes[2].set_title(f"{sname} Post − Pre")
    axes[2].set_xlabel(""); axes[2].set_ylabel("")

    cbar1 = fig.colorbar(im0, ax=axes[:2], shrink=0.9)
    cbar1.set_label("Std of SIC (masked)")

    cbar2 = fig.colorbar(im2, ax=axes[2], shrink=0.9)
    cbar2.set_label("Δ Std(SIC)")

    plt.savefig(f"{outdir}/std_SIC_mask015_{sname}_pre_post_diff.png", dpi=300)
    plt.close()

print("Done.")
