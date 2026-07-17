#!/usr/bin/env python
# ============================================================
# sigma_SIA: Standard Deviation of Daily SIC Anomaly Relative
# to Monthly Climatology (Vichi, 2022, TC 16, 4087-4106)
#
# This is NOT the same computation as monthly_sic_std.py.
# That script computed std of raw SIC values within a period.
# This script follows Vichi's actual method:
#
#   1. For each period (pre/post), compute a monthly
#      climatology from THAT PERIOD'S OWN data only (not the
#      full record). Using period-specific climatologies
#      means a genuine mean-timing shift (e.g. earlier melt)
#      does not get counted as "variability" -- it isolates
#      dispersion around each period's own expected seasonal
#      cycle, consistent with how this chapter treats
#      period-specific phase-date climatologies elsewhere.
#   2. Daily anomaly = daily SIC - that pixel's climatological
#      mean SIC for that calendar month, within that period.
#   3. sigma_SIA (pooled, per Vichi Eq. 3) = std of all daily
#      anomalies for a given calendar month, pooling across
#      all years in the period together (not averaging
#      per-year stds -- pooling first, per Vichi's method).
#
# Zero-SIC days are NOT excluded (see discussion from the
# monthly_sic_std.py run -- excluding them censors exactly the
# collapse-to-open-water swings that matter for this question).
# Land/permanently-missing pixels are masked; ice-zone mask
# (climatological max SIC >= 15%) restricts the summary to
# pixels that are ever meaningfully ice-covered.
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

outdir = Path("/user/geog/falejandraperez/sea-ice-phase/results/figures/sigma_SIA")
outdir.mkdir(parents=True, exist_ok=True)

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]

MAKE_SPATIAL_MAPS = False  # flip to True once the summary looks right
CMAP_MAIN = "magma_r"
CMAP_DIFF = "RdBu_r"
PCTL = 99

# ============================================================
# LOAD
# ============================================================
ds = xr.open_dataset(sic_file, chunks={"time": 365})
sic_raw = ds[sic_var]

land_mask = sic_raw.isnull().all("time")
ocean_mask = ~land_mask

sic = sic_raw.where(ocean_mask)
if float(sic.max()) > 1.5:
    sic = sic / 100.0

sic = sic.sel(time=~((sic.time.dt.month == 2) & (sic.time.dt.day == 29)))

ice_zone = sic.max("time", skipna=True) >= 0.15

sic_pre  = sic.sel(time=slice(*pre_period))
sic_post = sic.sel(time=slice(*post_period))


def sigma_SIA_by_month(sic_period, ice_zone):
    """
    Returns:
      summary: dict[month_name] -> circumpolar ice-zone mean sigma_SIA (float)
      spatial: dict[month_num] -> sigma_SIA DataArray (pixel-wise, pooled over years)
    """
    summary = {}
    spatial = {}

    for m in range(1, 13):
        sel = sic_period.time.dt.month == m
        sic_m = sic_period.sel(time=sel)  # all days in this calendar month, all years in period

        # period-specific monthly climatology: mean SIC per pixel
        # across all years, for this calendar month
        clim_m = sic_m.mean("time", skipna=True)

        # daily anomaly relative to that period's own climatology
        anomaly_m = sic_m - clim_m

        # pooled std across all days x years in this month (Vichi Eq 3)
        sigma_m = anomaly_m.std("time", skipna=True).compute()
        sigma_m = sigma_m.where(ice_zone)

        spatial[m] = sigma_m
        summary[MONTH_NAMES[m - 1]] = float(sigma_m.mean(skipna=True))

        print(f"  {MONTH_NAMES[m-1]}: sigma_SIA (ice-zone mean) = {summary[MONTH_NAMES[m-1]]:.4f}")

    return summary, spatial


# ============================================================
# PART 1 — SUMMARY: sigma_SIA by month, pre vs post
# ============================================================
print("Computing sigma_SIA for PRE period (own climatology)...")
summary_pre, spatial_pre = sigma_SIA_by_month(sic_pre, ice_zone)

print("Computing sigma_SIA for POST period (own climatology)...")
summary_post, spatial_post = sigma_SIA_by_month(sic_post, ice_zone)

pre_vals  = [summary_pre[m] for m in MONTH_NAMES]
post_vals = [summary_post[m] for m in MONTH_NAMES]

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(MONTH_NAMES, pre_vals, marker="o", label="1979-2016", color="#2E5395")
ax.plot(MONTH_NAMES, post_vals, marker="o", label="2017-2024", color="#C0392B")
ax.set_ylabel(r"$\sigma_{SIA}$ (ice zone mean)")
ax.set_title(r"Monthly $\sigma_{SIA}$: daily SIC anomaly vs. own-period monthly climatology"
             "\n(Vichi 2022 method), pre vs post 2016")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(outdir / "sigma_SIA_summary_pre_post.png", dpi=300)
plt.close()

print(f"\nSaved summary: {outdir / 'sigma_SIA_summary_pre_post.png'}")

pct_change = [100 * (post_vals[i] - pre_vals[i]) / pre_vals[i] for i in range(12)]
print("\n--- % change (post vs pre), sigma_SIA ---")
for name, pc in zip(MONTH_NAMES, pct_change):
    print(f"  {name}: {pc:+.1f}%")

# ============================================================
# PART 2 — SPATIAL MAPS (optional)
# ============================================================
if MAKE_SPATIAL_MAPS:
    print("\nComputing spatial sigma_SIA maps...")

    all_vals = (np.concatenate([spatial_pre[m].values.ravel() for m in range(1, 13)]))
    vmax_main = np.nanpercentile(all_vals, PCTL)

    diffs = {m: (spatial_post[m] - spatial_pre[m]) for m in range(1, 13)}
    vmax_diff = np.nanpercentile(
        np.abs(np.concatenate([diffs[m].values.ravel() for m in range(1, 13)])), PCTL
    )

    for m in range(1, 13):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

        spatial_pre[m].plot(ax=axes[0], cmap=CMAP_MAIN, vmin=0, vmax=vmax_main, add_colorbar=True)
        axes[0].set_title(f"{MONTH_NAMES[m-1]} sigma_SIA Pre")

        spatial_post[m].plot(ax=axes[1], cmap=CMAP_MAIN, vmin=0, vmax=vmax_main, add_colorbar=True)
        axes[1].set_title(f"{MONTH_NAMES[m-1]} sigma_SIA Post")

        diffs[m].plot(ax=axes[2], cmap=CMAP_DIFF, vmin=-vmax_diff, vmax=vmax_diff, add_colorbar=True)
        axes[2].set_title(f"{MONTH_NAMES[m-1]} Post - Pre")

        for ax in axes:
            ax.set_xlabel(""); ax.set_ylabel("")

        out = outdir / f"sigma_SIA_{MONTH_NAMES[m-1]}_pre_post_diff.png"
        plt.savefig(out, dpi=300)
        plt.close()
        print(f"  Saved {out}")

print("\nDone.")