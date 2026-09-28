"""
compute_sia.py

Computes daily, per-sector sea ice area (SIA) from the merged Bootstrap SIC
record, using the exact NSIDC-0771 cell-area grid and the canonical sector
mask. Flags days with suspiciously low valid-pixel coverage (e.g. the
1987/1991/1995 sensor-transition periods) rather than silently treating
them as real SIA=0 observations.

Run this AFTER re-running merge_smmr_patched.py and rebuild_merged_patched.py,
so the corrupted-to-zero NaN issue is fixed at the source.
"""

import xarray as xr
import numpy as np
import pandas as pd

MERGED_FILE  = "/user/geog/falejandraperez/sea-ice-phase/data/merged/merged_bootstrap_SH_latest.nc"
AREA_FILE    = "/user/geog/falejandraperez/sea-ice-phase/data/NSIDC0771_CellArea_PS_S25km_v1.1.nc"
SECTORS_FILE = "/user/geog/falejandraperez/sea-ice-phase/data/canonical_sectors.nc"
OUT_CSV      = "/user/geog/falejandraperez/sea-ice-phase/data/merged/sia_by_sector_daily.csv"

MIN_VALID_COVERAGE = 0.5  # fraction of sector's ocean pixels that must have
                           # valid (non-NaN) SIC for that day's SIA to be trusted

SECTOR_NAMES = {
    1: "Amundsen-Bellingshausen",
    2: "Weddell",
    3: "King Haakon VII",
    4: "East Antarctica",
    5: "Ross-Amundsen",
}

print("Loading files...")
ds = xr.open_dataset(MERGED_FILE)
area_ds = xr.open_dataset(AREA_FILE)
sectors_ds = xr.open_dataset(SECTORS_FILE)

# sanity check grids align before doing anything else
assert np.array_equal(area_ds.x.values, ds.x.values), "x grid mismatch: area vs SIC"
assert np.array_equal(area_ds.y.values, ds.y.values), "y grid mismatch: area vs SIC"
assert np.array_equal(sectors_ds.x.values, ds.x.values), "x grid mismatch: sectors vs SIC"
assert np.array_equal(sectors_ds.y.values, ds.y.values), "y grid mismatch: sectors vs SIC"
print("Grid alignment OK.")

sic = ds["N07_ICECON"]                       # (time, y, x)
cell_area_km2 = area_ds["cell_area"] / 1e6   # m^2 -> km^2
sector_id = sectors_ds["sector_id"]          # (y, x)
valid_ocean = sectors_ds["valid_ocean"]      # (y, x) bool

# --- mask out land/flag values (SIC should only be 0-1) and non-ocean pixels ---
sic_valid = sic.where(sic <= 1.0).where(valid_ocean)

# quick check that NaN is actually being used for missing data in this file
n_nan = int(np.isnan(sic_valid.values).sum())
print(f"NaN count in masked SIC array: {n_nan} "
      f"(should be > 0 - if this is 0 across the whole record, investigate "
      f"before trusting results)")

sia_records = {}
coverage_records = {}

for sid, name in SECTOR_NAMES.items():
    mask = (sector_id == sid) & valid_ocean
    n_ocean_pixels = int(mask.sum().item())

    sic_sector = sic_valid.where(mask)

    # SIA: sum of SIC * cell area over the sector, each day
    sia = (sic_sector * cell_area_km2).sum(dim=["x", "y"], skipna=True)

    # coverage: fraction of this sector's ocean pixels that had valid
    # (non-NaN) SIC on that day - low values mean the day is untrustworthy,
    # not that ice was actually absent
    obs_count = (~np.isnan(sic_sector)).sum(dim=["x", "y"])
    coverage_frac = obs_count / n_ocean_pixels

    sia_records[name] = sia.values
    coverage_records[f"{name}_coverage"] = coverage_frac.values

    print(f"{name}: {n_ocean_pixels} ocean pixels, "
          f"mean SIA={float(sia.mean()):,.0f} km^2, "
          f"days below {MIN_VALID_COVERAGE:.0%} coverage="
          f"{int((coverage_frac.values < MIN_VALID_COVERAGE).sum())}")

sia_df = pd.DataFrame(sia_records, index=pd.to_datetime(ds.time.values))
coverage_df = pd.DataFrame(coverage_records, index=pd.to_datetime(ds.time.values))
full_df = pd.concat([sia_df, coverage_df], axis=1)
full_df.index.name = "date"

# flag any day where ANY sector fell below the coverage threshold
low_cov_cols = [c for c in full_df.columns if c.endswith("_coverage")]
full_df["any_sector_low_coverage"] = (full_df[low_cov_cols] < MIN_VALID_COVERAGE).any(axis=1)

n_flagged = int(full_df["any_sector_low_coverage"].sum())
print(f"\n{n_flagged} / {len(full_df)} days flagged as low-coverage "
      f"(recommend excluding these from forcing-response regression)")

print("\nYear distribution of flagged low-coverage days:")
print(full_df.loc[full_df["any_sector_low_coverage"], :].index.year.value_counts().sort_index())

full_df.to_csv(OUT_CSV)
print(f"\nSaved full daily per-sector SIA + coverage flags to:\n  {OUT_CSV}")
print("\nPreview:")
print(full_df.head())
print(full_df.describe())