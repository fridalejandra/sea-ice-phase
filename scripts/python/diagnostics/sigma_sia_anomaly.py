#!/usr/bin/env python
# ============================================================
# sigma_SIA, computed PER INDIVIDUAL YEAR then aggregated by
# MEDIAN across years (Vichi 2022's actual method, Section 2.2
# and Figs 2-3 -- NOT the pooled Eq. 3 version we ran before).
#
# Prior scripts (monthly_sic_std.py, sigma_SIA_monthly.py)
# pooled all days from all years in a period together before
# taking one std. That is mathematically shift-invariant and
# turned out to just be raw std in disguise.
#
# THIS script instead:
#   1. Computes a fixed reference climatology per pixel/month,
#      from that period's own years (removes trend/mean-shift
#      from counting as "variability", consistent with earlier
#      reasoning).
#   2. For EACH INDIVIDUAL YEAR within the period, computes
#      sigma_SIA for that single year/month as the RMS
#      deviation from the fixed reference (Vichi Eq 2) -- this
#      is NOT the same as that year's own internal std, because
#      the reference is fixed, not that year's own mean.
#   3. Aggregates across years within the period using the
#      MEDIAN (Vichi's approach, robust to the heavy right tail
#      he documents) rather than pooling first.
#
# This is expected to produce genuinely different numbers than
# the pooled version -- if it doesn't, something is still wrong
# and worth flagging before drawing conclusions.
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

outdir = Path("/user/geog/falejandraperez/sea-ice-phase/results/figures/sigma_SIA_peryear")
outdir.mkdir(parents=True, exist_ok=True)

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]

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


def sigma_SIA_per_year(sic_period, ice_zone):
    """
    Returns:
      median_summary: dict[month_name] -> circumpolar ice-zone
        mean of the per-pixel MEDIAN (across years) sigma_SIA
      mean_summary: same but using per-pixel MEAN across years,
        for comparison against the median
      n_years: number of years contributing
    """
    median_summary = {}
    mean_summary = {}
    years = np.unique(sic_period.time.dt.year.values)
    n_years = len(years)

    for m in range(1, 13):
        sel_month = sic_period.time.dt.month == m
        sic_m = sic_period.sel(time=sel_month)

        # fixed reference climatology for this period/month:
        # mean SIC per pixel across all years, this calendar month
        clim_m = sic_m.mean("time", skipna=True).compute()

        per_year_sigma = []  # will hold one DataArray per year

        for y in years:
            sic_y_m = sic_m.sel(time=sic_m.time.dt.year == y)
            if sic_y_m.sizes["time"] == 0:
                continue
            anomaly = sic_y_m - clim_m
            # RMS deviation from the FIXED reference, over this
            # single year's ~30 days (Vichi Eq 2, per-instance)
            rms_y = np.sqrt((anomaly ** 2).mean("time", skipna=True)).compute()
            per_year_sigma.append(rms_y)

        stacked = xr.concat(per_year_sigma, dim="year")

        median_map = stacked.median("year", skipna=True).where(ice_zone)
        mean_map   = stacked.mean("year", skipna=True).where(ice_zone)

        median_summary[MONTH_NAMES[m - 1]] = float(median_map.mean(skipna=True))
        mean_summary[MONTH_NAMES[m - 1]]   = float(mean_map.mean(skipna=True))

        print(f"  {MONTH_NAMES[m-1]}: median-across-years sigma_SIA = "
              f"{median_summary[MONTH_NAMES[m-1]]:.4f}  "
              f"(mean-across-years = {mean_summary[MONTH_NAMES[m-1]]:.4f})")

    return median_summary, mean_summary, n_years


print(f"Computing per-year sigma_SIA for PRE period...")
median_pre, mean_pre, n_pre = sigma_SIA_per_year(sic_pre, ice_zone)

print(f"\nComputing per-year sigma_SIA for POST period...")
median_post, mean_post, n_post = sigma_SIA_per_year(sic_post, ice_zone)

print(f"\n(n_years: pre={n_pre}, post={n_post})")

# ============================================================
# PLOT — median comparison (the actual Vichi-style metric)
# ============================================================
median_pre_vals  = [median_pre[m] for m in MONTH_NAMES]
median_post_vals = [median_post[m] for m in MONTH_NAMES]

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(MONTH_NAMES, median_pre_vals, marker="o", label=f"1979-2016 (n={n_pre})", color="#2E5395")
ax.plot(MONTH_NAMES, median_post_vals, marker="o", label=f"2017-2024 (n={n_post})", color="#C0392B")
ax.set_ylabel(r"Median-across-years $\sigma_{SIA}$")
ax.set_title("Monthly sigma_SIA, median across individual years\n(Vichi 2022 method)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(outdir / "sigma_SIA_median_pre_post.png", dpi=300)
plt.close()
print(f"\nSaved: {outdir / 'sigma_SIA_median_pre_post.png'}")

pct_change_median = [100 * (median_post_vals[i] - median_pre_vals[i]) / median_pre_vals[i]
                      for i in range(12)]
print("\n--- % change (post vs pre), MEDIAN-across-years sigma_SIA ---")
for name, pc in zip(MONTH_NAMES, pct_change_median):
    print(f"  {name}: {pc:+.1f}%")

# sanity check against the pooled version's known result, so
# it's obvious at a glance whether this is actually different
print("\n--- For comparison: pooled result from sigma_SIA_monthly.py was ---")
print("  Jan: -31.5%  Feb: -40.7%  Mar: -29.3%  Apr: -23.0%")
print("  May: -8.7%   Jun: -9.5%   Jul: -7.3%   Aug: -1.8%")
print("  Sep: -8.5%   Oct: -5.2%   Nov: -6.7%   Dec: -13.9%")

print("\nDone.")