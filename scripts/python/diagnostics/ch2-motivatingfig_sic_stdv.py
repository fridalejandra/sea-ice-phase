#!/usr/bin/env python
# ==========================================================
# Motivating Figure: Threshold Ambiguity in Fixed-15%-SIC
# Transition Detection
#
# Purpose: show WHERE a single fixed SIC threshold gives an
# unstable/ambiguous read on "transition timing" -- i.e. where
# SIC crosses 15% multiple times per season, or spends
# extended time hovering near the threshold. This is a
# baseline spatial characterization (full record), NOT a
# pre/post trend test -- it motivates why a two-method
# framework is needed at all, independent of whether ambiguity
# has changed over time.
#
# Two complementary metrics, both climatological (full record):
#   (1) Mean number of 15% SIC crossings per ice season
#       (upward + downward). A clean single transition each
#       way = 2 crossings/season. More than that indicates
#       oscillation around the threshold.
#   (2) Mean fraction of ice-season days spent within a +/-5%
#       band around the threshold (10-20% SIC) -- pixels that
#       hover near 15% for extended periods, even without
#       fully crossing back and forth.
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

outdir = Path("/user/geog/falejandraperez/sea-ice-phase/results/figures/motivating_fig")
outdir.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.15
BAND_HALF_WIDTH = 0.05   # 10-20% band around threshold
CMAP = "magma_r"
PCTL = 99  # robust color scaling

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

# ice-zone mask: pixel is ever meaningfully ice-covered
# (climatological max SIC >= threshold), consistent with the
# sentinel-mask logic used elsewhere in this chapter
ice_zone = sic.max("time", skipna=True) >= THRESHOLD

# ============================================================
# METRIC 1 — crossing frequency per ice-season-year
#
# An "ice season" here runs Feb 1 - Jan 31 to keep each
# Freeze->Melt cycle intact (Antarctic minimum ~Feb, so this
# avoids splitting a single season's crossings across the
# calendar-year boundary). Adjust if your chapter already
# defines ice-year boundaries differently elsewhere -- keep
# this consistent with the FS/MS detection scripts.
# ============================================================
print("Computing crossing frequency per ice-season-year...")

above = (sic >= THRESHOLD)

# ice-year label: Feb of year Y through Jan of year Y+1 = ice-year Y
month = sic["time"].dt.month
year = sic["time"].dt.year
ice_year = xr.where(month >= 2, year, year - 1)

crossing = above.astype(int).diff("time").fillna(0) != 0
crossing_by_year = crossing.assign_coords(ice_year=("time", ice_year.values[1:]))

n_crossings_per_year = crossing_by_year.groupby("ice_year").sum("time")
mean_crossings = n_crossings_per_year.mean("ice_year", skipna=True).compute()
mean_crossings = mean_crossings.where(ice_zone)

# ============================================================
# METRIC 2 — fraction of days in the ambiguous band
# ============================================================
print("Computing fraction of days in 10-20% band...")

in_band = ((sic >= THRESHOLD - BAND_HALF_WIDTH) &
           (sic <= THRESHOLD + BAND_HALF_WIDTH))

frac_in_band = in_band.mean("time", skipna=True).compute()
frac_in_band = frac_in_band.where(ice_zone)

# ============================================================
# PLOT
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

vmax1 = np.nanpercentile(mean_crossings.values, PCTL)
im0 = mean_crossings.plot(ax=axes[0], cmap=CMAP, vmin=0, vmax=vmax1, add_colorbar=True)
axes[0].set_title("Mean 15% SIC crossings per ice-season-year\n(clean single transition = 2)")
axes[0].set_xlabel(""); axes[0].set_ylabel("")

vmax2 = np.nanpercentile(frac_in_band.values, PCTL)
im1 = frac_in_band.plot(ax=axes[1], cmap=CMAP, vmin=0, vmax=vmax2, add_colorbar=True)
axes[1].set_title("Fraction of ice-season days with SIC in 10-20% band")
axes[1].set_xlabel(""); axes[1].set_ylabel("")

plt.savefig(outdir / "threshold_ambiguity_motivating_fig.png", dpi=300)
plt.close()

print(f"Saved: {outdir / 'threshold_ambiguity_motivating_fig.png'}")

# ============================================================
# QUICK NUMERIC SUMMARY — for judging "is this null or not"
# before you commit to using it as the motivating figure
# ============================================================
vals1 = mean_crossings.values
vals2 = frac_in_band.values

print("\n--- Summary (ice zone only) ---")
print(f"Mean crossings/year: median={np.nanmedian(vals1):.2f}, "
      f"90th pctl={np.nanpercentile(vals1, 90):.2f}, "
      f"max={np.nanmax(vals1):.2f}")
print(f"  Fraction of ice-zone pixels with >2 crossings/year "
      f"(i.e. more than one clean up+down): "
      f"{100*np.nanmean(vals1 > 2):.1f}%")
print(f"Fraction of days in 10-20% band: median={np.nanmedian(vals2)*100:.1f}%, "
      f"90th pctl={np.nanpercentile(vals2, 90)*100:.1f}%")

print("Done.")