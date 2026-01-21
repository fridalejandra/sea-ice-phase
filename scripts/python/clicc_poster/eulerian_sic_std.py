import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

sic_file = "/user/geog/falejandraperez/sea-ice-phase/data/merged/merged_bootstrap_SH_latest.nc"
sic_var  = "N07_ICECON"

# periods
pre_slice  = slice("1979-01-01", "2016-12-31")
post_slice = slice("2017-01-01", "2024-12-31")

# masking threshold
thr_open = 0.15

# plotting
cmap_main = "magma_r"      # low=light, high=dark
cmap_diff = "RdBu_r"

# seasons
seasons = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
}

def drop_feb29(da):
    return da.sel(time=~((da.time.dt.month == 2) & (da.time.dt.day == 29)))

def ensure_01(sic):
    return sic/100.0 if float(sic.max()) > 1.5 else sic

def season_select(da, months):
    return da.sel(time=da.time.dt.month.isin(months))

def robust_vmax(da, q=0.995):
    # ignore NaNs
    return float(da.quantile(q, skipna=True))

def compute_std_sic_change(sic_da, use_abs=False):
    dsic = sic_da.diff("time")
    if use_abs:
        dsic = np.abs(dsic)   # IMPORTANT: xarray DataArray has no .abs()
    return dsic.std("time", skipna=True)

def compute_std_sic(sic_da):
    return sic_da.std("time", skipna=True)

# -----------------------------
# LOAD
# -----------------------------
ds = xr.open_dataset(sic_file)
sic = ensure_01(drop_feb29(ds[sic_var]))

sic_pre  = sic.sel(time=pre_slice)
sic_post = sic.sel(time=post_slice)

# -----------------------------
# LOOP SEASONS
# -----------------------------
for sname, months in seasons.items():
    sp = season_select(sic_pre, months)
    so = season_select(sic_post, months)

    # ==========
    # FIG A: Std(dSIC) (signed)  [switch use_abs=True if you want |dSIC|]
    # ==========
    std_pre  = compute_std_sic_change(sp, use_abs=False)
    std_post = compute_std_sic_change(so, use_abs=False)
    std_diff = std_post - std_pre

    # robust scaling
    vmax = max(robust_vmax(std_pre), robust_vmax(std_post))
    vmax_diff = robust_vmax(np.abs(std_diff))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    im0 = std_pre.plot(ax=axes[0], cmap=cmap_main, vmin=0, vmax=vmax, add_colorbar=False, robust=False)
    axes[0].set_title(f"{sname} Std(ΔSIC) Pre")
    axes[0].set_xlabel(""); axes[0].set_ylabel("")

    im1 = std_post.plot(ax=axes[1], cmap=cmap_main, vmin=0, vmax=vmax, add_colorbar=False, robust=False)
    axes[1].set_title(f"{sname} Std(ΔSIC) Post")
    axes[1].set_xlabel(""); axes[1].set_ylabel("")

    im2 = std_diff.plot(ax=axes[2], cmap=cmap_diff, vmin=-vmax_diff, vmax=vmax_diff, add_colorbar=False, robust=False)
    axes[2].set_title(f"{sname} Post − Pre")
    axes[2].set_xlabel(""); axes[2].set_ylabel("")

    cbar1 = fig.colorbar(im0, ax=axes[:2], orientation="vertical", shrink=0.9)
    cbar1.set_label("Std of daily SIC change, std(ΔSIC)")

    cbar2 = fig.colorbar(im2, ax=axes[2], orientation="vertical", shrink=0.9)
    cbar2.set_label("Δ std(ΔSIC)")

    outA = f"/user/geog/falejandraperez/sea-ice-phase/results/figures/clicc_poster/std_dSIC_{sname}_pre_post_diff.png"
    plt.savefig(outA, dpi=300)
    plt.close()

    # ==========
    # FIG B: Std(SIC) with 0–15% excluded (mask open water / near-open water)
    # ==========
    std_pre  = compute_std_sic(sp)
    std_post = compute_std_sic(so)
    std_diff = std_post - std_pre

    # period-specific seasonal mean masks (pre/post separately)
    mask_pre  = sp.mean("time", skipna=True)  > thr_open
    mask_post = so.mean("time", skipna=True) > thr_open

    std_pre_m  = std_pre.where(mask_pre)
    std_post_m = std_post.where(mask_post)
    std_diff_m = std_post_m - std_pre_m

    vmax = max(robust_vmax(std_pre_m), robust_vmax(std_post_m))
    vmax_diff = robust_vmax(np.abs(std_diff_m))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    im0 = std_pre_m.plot(ax=axes[0], cmap=cmap_main, vmin=0, vmax=vmax, add_colorbar=False, robust=False)
    axes[0].set_title(f"{sname} Std(SIC) Pre (SIC>{thr_open})")
    axes[0].set_xlabel(""); axes[0].set_ylabel("")

    im1 = std_post_m.plot(ax=axes[1], cmap=cmap_main, vmin=0, vmax=vmax, add_colorbar=False, robust=False)
    axes[1].set_title(f"{sname} Std(SIC) Post (SIC>{thr_open})")
    axes[1].set_xlabel(""); axes[1].set_ylabel("")

    im2 = std_diff_m.plot(ax=axes[2], cmap=cmap_diff, vmin=-vmax_diff, vmax=vmax_diff, add_colorbar=False, robust=False)
    axes[2].set_title(f"{sname} Post − Pre")
    axes[2].set_xlabel(""); axes[2].set_ylabel("")

    cbar1 = fig.colorbar(im0, ax=axes[:2], orientation="vertical", shrink=0.9)
    cbar1.set_label("Std of SIC, std(SIC) (masked)")

    cbar2 = fig.colorbar(im2, ax=axes[2], orientation="vertical", shrink=0.9)
    cbar2.set_label("Δ std(SIC) (masked)")

    outB = f"/user/geog/falejandraperez/sea-ice-phase/results/figures/clicc_poster/std_SIC_mask015_{sname}_pre_post_diff.png"
    plt.savefig(outB, dpi=300)
    plt.close()

print("Done.")
