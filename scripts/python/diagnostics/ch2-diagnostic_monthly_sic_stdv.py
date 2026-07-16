#!/usr/bin/env python
# ============================================================
# Monthly SIC Variability, Pre vs Post 2016
#
# Consolidates and fixes inconsistencies found across
# eulerian_sic_std.py / JJA_Std.py / std_daily_sic.py:
#
#   - Does NOT exclude SIC == 0 days. Excluding them (as
#     eulerian_sic_std.py did via `sp > 0`) censors exactly
#     the collapse-to-open-water swings that matter most for
#     a variability question. Only land / permanently-missing
#     pixels are masked, following JJA_Std.py's approach.
#   - Uses percentile-based (robust) color scaling instead of
#     raw max(), so a single noisy pixel can't compress the
#     color range (std_daily_sic.py's approach, applied
#     consistently here).
#   - Computes std per CALENDAR MONTH (12 groups) rather than
#     per season (4 groups), pre vs post 2016.
#   - Runs a cheap circumpolar/per-sector summary FIRST
#     (fast) before optionally generating full spatial maps
#     for all 12 months (slow) — set MAKE_SPATIAL_MAPS=True
#     once the summary shows something worth chasing.
# ============================================================

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# SETTINGS
# ============================================================
sic_file = "/user/geog/falejandraperez/sea-ice-phase/data/merged/merged_bootstrap_SH_latest.nc"
sic_var  = "N07_ICECON"

pre_period  = ("1979-01-01", "2016-12-31")
post_period = ("2017-01-01", "2024-12-31")

outdir = Path("/user/geog/falejandraperez/sea-ice-phase/results/figures/monthly_std")
outdir.mkdir(parents=True, exist_ok=True)

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]

MAKE_SPATIAL_MAPS = False   # flip to True after checking the summary plot
CMAP_MAIN = "magma_r"
CMAP_DIFF = "RdBu_r"
PCTL = 99  # robust color scaling percentile

# Optional: restrict the summary to sectors, e.g.
#   {"Weddell": slice(...), "Ross-Amundsen": slice(...)}
# Left as None = circumpolar ice-zone average only. Add sector
# slices/masks here if you want the summary split regionally
# before committing to spatial maps.
SECTORS = None

# ============================================================
# LOAD
# ============================================================
ds = xr.open_dataset(sic_file, chunks={"time": 365})
sic_raw = ds[sic_var]

# land = permanently missing across the whole record
land_mask = sic_raw.isnull().all("time")
ocean_mask = ~land_mask

sic = sic_raw.where(ocean_mask)
if float(sic.max()) > 1.5:
    sic = sic / 100.0

# drop Feb 29 for calendar consistency across years
sic = sic.sel(time=~((sic.time.dt.month == 2) & (sic.time.dt.day == 29)))

# ice-zone mask for the summary: pixel is ever meaningfully
# ice-covered at some point in the full record (climatological
# max SIC >= 15%), consistent with the sentinel-mask logic
# used elsewhere in this chapter
ice_zone = sic.max("time", skipna=True) >= 0.15

sic_pre  = sic.sel(time=slice(*pre_period))
sic_post = sic.sel(time=slice(*post_period))

# ============================================================
# PART 1 — CHEAP SUMMARY: mean std by month, pre vs post
# ============================================================
print("Computing monthly summary (ice-zone mean std, pre vs post)...")

summary_pre, summary_post = [], []

for m in range(1, 13):
    pre_m  = sic_pre.sel(time=sic_pre.time.dt.month == m)
    post_m = sic_post.sel(time=sic_post.time.dt.month == m)

    std_pre_m  = pre_m.std("time", skipna=True).where(ice_zone)
    std_post_m = post_m.std("time", skipna=True).where(ice_zone)

    summary_pre.append(float(std_pre_m.mean(skipna=True)))
    summary_post.append(float(std_post_m.mean(skipna=True)))
    print(f"  {MONTH_NAMES[m-1]}: pre={summary_pre[-1]:.4f}  post={summary_post[-1]:.4f}")

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(MONTH_NAMES, summary_pre, marker="o", label="1979–2016", color="#2E5395")
ax.plot(MONTH_NAMES, summary_post, marker="o", label="2017–2024", color="#C0392B")
ax.set_ylabel("Mean Std(SIC), ice zone")
ax.set_title("Monthly SIC variability, pre vs post 2016 (circumpolar ice zone)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(outdir / "monthly_std_summary_pre_post.png", dpi=300)
plt.close()

print(f"Saved summary: {outdir / 'monthly_std_summary_pre_post.png'}")

# ============================================================
# PART 2 — SPATIAL MAPS (optional, per month)
# ============================================================
if MAKE_SPATIAL_MAPS:
    print("Computing full spatial monthly maps (this will take longer)...")

    # gather all diffs first to set one shared, robust color scale
    diffs = []
    per_month_data = {}

    for m in range(1, 13):
        pre_m  = sic_pre.sel(time=sic_pre.time.dt.month == m)
        post_m = sic_post.sel(time=sic_post.time.dt.month == m)

        std_pre_m  = pre_m.std("time", skipna=True).compute()
        std_post_m = post_m.std("time", skipna=True).compute()
        std_diff_m = (std_post_m - std_pre_m).compute()

        per_month_data[m] = (std_pre_m, std_post_m, std_diff_m)
        diffs.append(std_diff_m)

    vmax_main = np.nanpercentile(
        np.concatenate([d[0].values.ravel() for d in per_month_data.values()] +
                        [d[1].values.ravel() for d in per_month_data.values()]),
        PCTL
    )
    vmax_diff = np.nanpercentile(
        np.abs(np.concatenate([d.values.ravel() for d in diffs])), PCTL
    )

    for m in range(1, 13):
        std_pre_m, std_post_m, std_diff_m = per_month_data[m]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

        std_pre_m.plot(ax=axes[0], cmap=CMAP_MAIN, vmin=0, vmax=vmax_main, add_colorbar=True)
        axes[0].set_title(f"{MONTH_NAMES[m-1]} Std(SIC) Pre")

        std_post_m.plot(ax=axes[1], cmap=CMAP_MAIN, vmin=0, vmax=vmax_main, add_colorbar=True)
        axes[1].set_title(f"{MONTH_NAMES[m-1]} Std(SIC) Post")

        std_diff_m.plot(ax=axes[2], cmap=CMAP_DIFF, vmin=-vmax_diff, vmax=vmax_diff, add_colorbar=True)
        axes[2].set_title(f"{MONTH_NAMES[m-1]} Post − Pre")

        for ax in axes:
            ax.set_xlabel("")
            ax.set_ylabel("")

        out = outdir / f"SIC_std_{MONTH_NAMES[m-1]}_pre_post_diff.png"
        plt.savefig(out, dpi=300)
        plt.close()
        print(f"  Saved {out}")

print("Done.")